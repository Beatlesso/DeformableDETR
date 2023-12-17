# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        # 按照 d_model 切分成多个头
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        # 用于CUDA实现的一个参数
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 定义四个需要用到的线性层
        # 采样点的坐标偏移，每个query在每个注意力头和每个特征层都需要采样n_points个
        # 由于x、y坐标都有偏移，因此还要乘以2
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 每个query对应的所有采样点的权重
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 线性变换得到value
        self.value_proj = nn.Linear(d_model, d_model)
        # 最后经过这个线性变换得到输出
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        '''
        初始化偏移预测的偏置(bias)，使得初始偏移位置犹如大小不同的方形卷积组合
        '''
        # 该函数用val的值填充输入的张量或变量， 即初始化变量为0
        constant_(self.sampling_offsets.weight.data, 0.)
        # 每个头对应着一个方向
        # (8, ) [0, 2pi / 8, 4pi / 8, ... , 14pi / 8]
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # (8, 2) 对8个方向每个方向取对应的余弦值和正弦值
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init / grid_init.abs().max(-1, keepdim=True)[0] 这一步计算得到 8 个头的对应坐标偏移
        # (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
        # 然后重复到所有特征层和采样点 (n_heads, n_levels, n_points, 2)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)

        # 同一特征层的不同采样点的坐标偏移肯定不能一样，因此这里做了处理
        # 比如说第一个头某个层的4个采样点，就由
        # (1, 0),(1, 0),(1, 0),(1, 0) 变成了 (1, 0),(2, 0),(3, 0),(4, 0) 
        # 从视觉上来看，形成的偏移位置相当于是 3x3、5x5、7x7、9x9的正方形卷积核（除去中心，中心是参考点本身）
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 这里取消了梯度，只是借助nn.Parameter把所有点的偏移放进去
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        # 按照论文描述，应该是
        # constant_(self.attention_weights.bias.data, 1. / (self.n_levels * self.n_points))

        # xavier_uniform_: 用一个均匀分布生成值，填充输入的张量或变量 结果张量中的值采样自 U(-a, a)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)  N是批次大小, Length_{query}是query的个数, query事实上就是加上了 pos_embed 的 input_flatten 

        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
         (bs, hw*lvl, lvl, 2)              or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes  代表参照点的位置

        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)  形状和 query 一样，说明  Length_{query} = \sum_{l=0}^{L-1} H_l \cdot W_l

        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]     每个特征图的尺寸

        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                                            是input_flatten中各特征层起始的index （这个索引对应到被flatten的维度）

        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        '''
            Multi-Scale Deformable Attention 主要做以下事情：
            1) 将输入input_flatten (公式中的x) (对于encoder就是backbone输出的特征图变换而来; 对于decoder就是encoder的输出) 
                通过变换矩阵得到 value, 同时将padding部分用0填充
            2) 将query (object query)
                (
                    对于encoder就是 特征图本身加上position & scale-level embedding; 
                    对于decoder就是 self-attention的输出加上position embedding;
                    2-stage时这个 position embedding是由encoder预测的top-k proposal boxes进行position embedding得来;
                    而1-stage是由预设的的embedding
                )
                分别通过两个全连接层来得到采样点对应的偏移和注意力权重(注意力权重会进行归一化)
            3) 根据参考点 (reference points: 
                            对于decoder来说, 2-stage时是encoder预测的top-k proposal boxes
                            1-stage时是由预设的query embedding经过全连接层得到, 两种情况都经过sigmoid函数归一化
                            而对于encoder来说, 就是各特征点在所有特征层对应的归一化中心坐标
                         ) 
                         坐标和预测的坐标偏移来得到采样点坐标
            4) 由采样点坐标在value中插值采样出对应的特征向量, 然后施加注意力权重, 最后将这个结果经过全连接层得到输出结果
        '''
        # 事实上在Encoder阶段 Len_q 和 Len_in一样
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # 判断所有特征层的点数之和是否为 Len_in
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # (N, Len_in, d_model)   将input_flatten经过线性层投影得到 value
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            # 将原图padding部分用0填充
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # (N, Len_in, n_heads, d_model / n_heads) 拆分成注意力头
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # (N, Len_q, n_heads, n_levels, n_points, 2) 预测采样点的坐标偏移
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # (N, Len_q, n_heads, n_levels * n_points) 由query计算注意力, 并且进行归一化
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # 归一化之后变回 (N, Len_q, n_heads, n_levels, n_points) 的形状
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # 计算采样点坐标 reference_points(N, Len_q, n_levels, 2 or 4)
        if reference_points.shape[-1] == 2:
            # (n_levels, 2) 将input_spatial_shapes的顺序反转一下, 每个点变成(W, H) 放到 offset_normalizer
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # reference_points 变成  (N, Len_q, 1, n_levels, 1, 2)
            # offset_normalizer 变成 (1, 1, 1, n_levels, 1, 2)
            # sampling_offsets       (N, Len_q, n_heads, n_levels, n_points, 2)
            # 即对坐标偏移量使用对应特征图的宽高进行归一化然后加在参考点坐标上得到归一化后的采样点坐标
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            # reference_points 的最后一维中前两个是中心坐标 xy, 后两个是宽高wh
            # 初始化的时候 offset的值在[-k, k]的范围(k = n_points) 因此这里除以 n_points相当于归一化到 0~1
            # 然后乘以宽和高的一半, 加上参考点中心坐标, 这样就使得偏移后的采样点位于 proposal bbox内
            # 相当于对采样范围进行了约束, 减小了搜索空间
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        # 根据采样点位置拿出对应的value, 并且施加预测出来的注意力权重 (和value进行weighted sum)
        # (N, Len_in, C)
        # 注: 实际调用的是基于CUDA实现的版本, 需要编译
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        # (N, Len_in, C)
        output = self.output_proj(output)
        return output
