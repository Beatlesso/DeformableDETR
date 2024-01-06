# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        # 双阶段建议区域数
        self.two_stage_num_proposals = two_stage_num_proposals

        # 创建编码层
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        # 将若干个编码层组合成编码器
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        # 创建解码层
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        # 将若干个解码层组合成解码器
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        # level_embed
        # 对每个特征层增加一个 d_model 维的embedding
        # 目的是为了区分 拥相同的(h,w)坐标但位于不同特征层的特征点
        # 随机初始化并且是随网络一起训练的、可学习的
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # 判断是否为双阶段的模式，单阶段和双阶段模式有些差别
        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    # 用于初始化参数
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules(): # modules()会递归地遍历所有子模块
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        # 随机初始化 level_embed
        normal_(self.level_embed)

    # 获取proposal位置编码
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    # 用Encoder的输出（memory）来得到proposals
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        # 计算每个尺寸的特征图上的proposal
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 取出当前特征图的mask，并且view成  (N_, H_, W_, 1)  的形状
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            # 计算每个样本上非padding的行和列数
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            # 生成两个网格，分别看作行和列，在另一个的维度上进行广播复制，得到两个 (H, W)大小的网格
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            # 将两个网格扩展出第三个维度 (H, W, 1)，并在第三个维度上cat (H, W, 2)，这样就得到了左上 (0, 0) 到右下 (H - 1, W - 1)的网格坐标了
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            # 将valid_W和valid_H拼接合并
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            # 首先将网格复制扩展成 (N, H, W, 2) 并且 + 0.5   然后再 / scale，相当于对网格坐标进行归一化
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # 获得一个形为 grid，值全为 2^lvl / 20 的张量wh
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # 将 grid 和 wh 在最后一个维度上进行拼接得到 (N, H, W, 4)的proposal -> (N, H*W, 4)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            # 添加到列表里
            proposals.append(proposal)
            # 对应下标增加
            _cur += (H_ * W_)
        
        # 将所有的proposal按维度1进行拼接，得到 (N, Len_q, 4) 的output_proposals
        output_proposals = torch.cat(proposals, 1)
        # 检查output_proposals最后一维上的所有行是否都满足 (output_proposals > 0.01) & (output_proposals < 0.99)
        # output_proposals_valid 是一个 (N, Len_q, 1) 的bool张量
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # 
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # masked_fill：在mask为True的情况下，用值填充自张量的元素。掩码的形状必须与底层张量的形状一起可广播。
        # 把 memory_padding_mask 以及 不合法的 output_proposal 用 inf填充
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        # 对output_memory 则对应用 0 进行填充
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        # 对 output_memory 进行线性层投影和层归一化
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    # 
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        # (bs, H) 在dim1上求sum => (bs,)
        valid_H = torch.sum(~mask[:, :, 0], 1)
        # (bs, W) 在dim1上求sum => (bs,)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        # (bs, 2)
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):

        # 如果是2-stage模式，那么Decoder的query embedding由Encoder预测的top-k proposal boxes进行位置嵌入来产生
        # 否则，使用预设的query_embed
        # query_embed 是一个包含 num_queries=300 个 hidden_dim*2 大小张量的查找表
        assert self.two_stage or query_embed is not None
        '''
        srcs (lvl, bs, C, H, W)
        masks (lvl, bs, H, W)
        pos_embeds (lvl, bs, C, H, W)
        query_embed ()

        为Encoder的输入做准备:
        i). 将各层特征图（已经映射到c=256维度）flatten并concat到一起: (bs, h1w1 + ... + hl*wl, c=256)
        ii). 将各层特征图对应的mask（指示了哪些位置是padding） flatten并concat：(bs, h1w1 + ... + hl*wl)
        iii). 将各层特征图对应的position embedding加上scale level embedding（用于表明query属于哪个特征层），然后flatten并concat：(bs, h1w1 + ... + hl*wl, c=256)
        iv). 将各层特征图的宽高由list变为tensor:  (lvl, 2)
        v). 由于将所有特征层的特征点concat在了一起，因此为了区分各层，需要计算对应于被flatten的那个维度的起始index（第一层当然是0，后面就是累加各层的）
        vi). 计算各特征层中非padding部分的边长（高&宽）占特征图边长的比例
        '''
        # prepare input for encoder
        # 以下的flatten是指将h, w两个维度展平为 h*w
        src_flatten = []
        mask_flatten = []
        # 各层特征图对应的position embedding + scale level embedding
        lvl_pos_embed_flatten = []
        # 各层特征图的尺寸
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape  # (bs, C, H, W)
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # (bs, C, H, W) => (bs, HW, C)
            src = src.flatten(2).transpose(1, 2)
            src_flatten.append(src)
            # (bs, H, W) => (bs, HW)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
            '''
            由于position embedding只区份 h,w 的位置
            因此对于不同特征层由相同坐标值的特征点来说，是无法区分的，因此要加上scale level embedding
            这样所有特征点的位置信息就各不相同了
            '''
            # (bs, C, H, W) => (bs, HW, C)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)

            # self.level_embed (lvl, C) => [lvl](1, 1, C) 
            # 即在实际使用的时候，每个lvl加相同的level_embed，不同lvl加不同level_embed
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # (bs, HW, C)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            
        # (bs, H1W1 + H2W2 + ... HlWl, C)    
        src_flatten = torch.cat(src_flatten, 1)
        # (bs, H1W1 + H2W2 + ... HlWl)   
        mask_flatten = torch.cat(mask_flatten, 1)
        # (bs, H1W1 + H2W2 + ... HlWl, C) 
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (lvl, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # .prod(dim=1) 是将dim1的各个元素相乘，这里得到各个特征层对应的特征点数 h*w
        # .consum(dim=0) 表示在dim0进行累加，这样就会得到 h1w1 + ... + hl*wl
        # .new_zeros 返回用 0 填充的大小为 size 的张量, 其可以方便的复制原来tensor的所有类型，比如数据类型和数据所在设备等等
        # 因此这里得到的 level_start_index 是各特征层起始的index （这个索引对应到被flatten的维度）
        # (1, ) cat (lvl-1, ) => (lvl, )
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # (bs, lvl, 2) 各特征层中非padding部分的边长（高&宽）占特征图边长的比例
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder   lvl_pos_embed_flatten 这个就是传入Encoder layer的的pos          mask_flatten就是传入Encoder layer的 mask
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # memory (N, Len_q, C)



        # prepare input for decoder
        bs, _, c = memory.shape
        # 根据是否 2-stage 分情况进行处理，因为生成的 reference points不同
        if self.two_stage:
            # 生成proposals，并且对Encoder的输出（memory）进行处理
            # (N, Len_q, C), (N, Len_q, 4)    其中proposal每个都是 xywh 的形式，并且是经过 inverse-sigmoid 函数后的结果
            # 其实这里的 output_proposals 对应的就是各层特征图各个特征点的位置（相当于anchor的形式，是固定的）
            # 因此还需要借助Decoder最后一层的bbox head来预测一个offset来获得一个更加灵活的结果
            # 这才是第一个阶段预测的 proposal boxes
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            # 注意这里对应的是多分类，并非二分类！
            # (N, Len_q, n_classes)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            # bbox 预测的是相对 proposal 的偏移，因此这里要相加，后续还要经过 sigmoid 函数才能得到bbox预测结果（归一化形式）
            # (N, Len_q, 4)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            # two_stage_num_proposals = 300
            topk = self.two_stage_num_proposals
            '''
            当不使用iterative bbox refine时，所有的class_embed参数共享
            这样会使得第二阶段对解码输出进行分类时都倾向于向第一个类别，貌似不妥
            '''
            # 取enc_outputs_class所有 N * Len_q 的第0个n_classes
            # torch.topk 返回沿给定维度的给定输入张量的k个最大元素以及对应索引
            # 选取得分最高的top-k分类预测，最后的[1]代表取得反回top-k对应的索引  (N, topk)
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]


            # 拿出top-k得分最高对应的预测 bbox：(N, topk, 4)
            '''
            torch.gather(input, dim, index, *, sparse_grad=False, out=None) → Tensor
                >>> t = torch.tensor([[1, 2], [3, 4]])
                >>> torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
                tensor([[ 1,  1],
                        [ 4,  3]])
            '''
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            # 取消梯度
            topk_coords_unact = topk_coords_unact.detach()
            # 经过 sigmoid，变成归一化形式，这个结果会送到Decoder中作为初始的bboxes估计
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            '''
            生成Decoder的query（target）和query embedding：
            对top-k proposal boxes进行位置编码，编码方式是给xywh每个都赋予128维
            其中每128维中，偶数维度用sin函数，奇数维度用cos函数编码
            然后经过全连接层和层归一化处理
            最终，前256维结果（对应xy位置）作为Decoder的query embedding（因为xy代表的是位置信息）
            后256维结果（对应wh）作为target（query）
            '''
            # (N, topk=300, 4*128=512)
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            # (N, topk=300, 256), (N, topk=300, 256)
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # 仅为与2-stage的情况兼容  query_embed (num_queries=300, C*2)
            # (n_query=300, 256), (n_query=300, 256)
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            # (N=bs, n_query=300, 256)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            # (N=bs, n_query=300, 256)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # 通过全连接层生成proposal参考点的sigmoid归一化坐标(cx, cy): (N, n_query=300, 2)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self_attention 注意这里用的是MSDeformAttn
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        # 只要有position embedding，就加上
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention  这边的reference_points就是 (bs, hw*lvl, lvl, 2) 的
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # 深度拷贝num_layers份encoder_layer，生成ModuleList
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # valid_ratios (bs, lvl, 2) 各特征层中非padding部分的边长（高&宽）占特征图边长的比例
        # spatial_shapes (lvl, 2) 各特征层的特征图尺寸
        # 获得参照点
        reference_points_list = []
        # 枚举每个 lvl 的尺寸
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 为当前 lvl 创建参照点网格， torch.linspace(a, b, c) 生成[a, b]的c个数构成的等差数列，这里生成 h*w个点
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # 首先将ref_y压成一维(hw,) 然后变成 (1, hw)
            # valid_ratios 取第lvl层，得到 (bs, 1) 最后做除法得到 (bs, hw)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # (bs, hw, 2)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # reference_points_list (lvl, bs, hw, 2) => reference_points(bs, hw*lvl, 2)
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points[:, :, None] (bs, hw*lvl, 1, 2)
        # valid_ratios[:, None] (bs, 1, lvl, 2)
        # 最终 reference_points (bs, hw*lvl, lvl, 2) 归一化到了 (0, 1) 之间 两个点是 (W, H)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    # 这里传进来的所有参数就是 Encoder准备工作得到的，然后一层一层前向传播即可
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        # 获取参照点
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod # tensor 与位置编码 合并
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention   q和k都是由tgt和query_pos加和得到的，v就是tgt本身
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        # 残差连接
        tgt = tgt + self.dropout2(tgt2)
        # 层归一化
        tgt = self.norm2(tgt)

        # cross attention   这里query是由tgt和query_pos加和得到的，src是input_flatten
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        # 残差连接
        tgt = tgt + self.dropout1(tgt2)
        # 层归一化
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        # 下面两个成员是在deformable_detr代码中赋值的
        self.bbox_embed = None
        self.class_embed = None

    # src是input_flatten，  src_valid_ratios是各特征层中非padding部分的边长（高&宽）占特征图边长的比例
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        # 枚举解码器的每一层
        for lid, layer in enumerate(self.layers):
            # 最后一维是4代表 two-stage模式
            if reference_points.shape[-1] == 4:
                # 将reference_point的四个坐标都 缩放到非padding的比例内
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else: # refine模式
                assert reference_points.shape[-1] == 2
                # 将reference_point 缩放到非padding的比例内
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            # 解码器单层计算
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


# 深度拷贝原始 module 并生成 ModuleList
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# 解系给定参数构建DeformableTransformer
def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


