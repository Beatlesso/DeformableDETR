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

import time
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
# 性能测试使用
from pyinstrument import Profiler

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output
    

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None



# class PyTorchDeformAttnFunction(Function):
#     # pytorch实现
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
#         '''
#             性能测试代码
#         '''
#         # profiler = Profiler()
#         # profiler.start()
#         '''
#             性能测试代码
#         '''
#         # for debug and test only,
#         # need to use cuda version instead
#         # (N, Len_in, n_heads, d_model / n_heads)     Len_in = H * W * n_level = len_q
#         bch, Len_in, n_heads, D_ = value.shape
#         # (N, Len_q, n_heads, n_levels, n_points, 2)
#         _, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
#         # 分割得到特征层的value  是一个list, list的每个对应元素i, dim=1的维度大小为 H_i * W_i 
#         # (n_level, N, H*W, n_heads, d_model / n_heads)
#         value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
#         # 由于以下使用了 F.grid_sample(), 要求采样位置的坐标归一化到[-1, 1]范围 ((-1, -1)代表左上角; (1, 1)代表右下角)
#         # 因此, 这里是将[0, 1]映射到[-1, 1]
#         sampling_grids = 2 * sampling_locations - 1
#         sampling_value_list = []
#         for lid_, (H_, W_) in enumerate(value_spatial_shapes):
#             # bch, H_*W_, n_heads, D_    ->    bch, H_*W_, n_heads*D_    ->    bch, n_heads*D_, H_*W_ -> bch*n_heads, D_, H_, W_
#             value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(bch*n_heads, D_, H_, W_)
#             # bch, Len_q, n_heads, n_points, 2    ->    bch, n_heads, Len_q, n_points, 2    ->    bch*n_heads, Len_q, n_points, 2
#             sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
#             # 根据采样点坐标在value中插值出对应的特征
#             # ps: grid_sample() 用于在归一化后的坐标对特征图进行采样 input是输入特征图，grid是归一化后的采样点，mode是插值方法
#             # rid_sample(input, grid, mode='bilinear', padding_mode='border', align_corners=True)
#             # 这里value_l充当被插值的特征图, 是input, 维度需要是4D/5D   (N, C, H, W)
#             # sampling_grid_l_则代表采样的位置, 是grid, 最后一维2对应input中的坐标(可能是小数)  (N, H, W, 2)
#             # 倒数第二、三维代表采样后输出特征图的宽、高
#             # input和grid的第一个维度必须一致，最终输出的通道数与input一致，是不变的
#             # bch*n_heads, D_, Len_q, n_points
#             '''
#             F.grid_sample 
#                 input(Tensor) -> 4D (N, C, H_in, W_in)    or 5D (N, C, D_in, H_in, W_in)
#                 grid(Tensor)  -> 4D (N, H_out, W_out, 2)  or 5D (N, D_out, H_out, W_out, 3)
#                 output(Tensor)-> 4D (N, C, H_out, W_out)
#             '''
#             sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
#                                             mode='bilinear', padding_mode='zeros', align_corners=False)
#             sampling_value_list.append(sampling_value_l_)

#         # (bch, Len_q, n_heads, n_levels, n_points)    ->    (bch, n_heads, Len_q, n_levels, n_points)    ->    (bch*n_heads, 1, Len_q, n_levels*n_points)
#         attention_weights = attention_weights.transpose(1, 2).reshape(bch*n_heads, 1, Len_q, n_levels*n_points)
#         # 将注意力权重和采样特征进行weighted sum
#         # (bch*n_heads, D_, Len_q, n_points)    ->    (bch*n_heads, D_, Len_q, n_level, n_points)    ->    (bch*n_heads, D_, Len_q, n_level * n_points)
        
#         output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bch, n_heads*D_, Len_q)

#         '''
#             性能测试代码
#         '''
#         # profiler.stop()
#         # profiler.print()

#         '''
#             性能测试代码
#         '''
    
#         # (bch, Len_q, hidden_dim)
#         return output.transpose(1, 2).contiguous()



# 打点测试代码
class PyTorchDeformAttnFunction(Function):
    # pytorch实现
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):

        start1 = time.time()
        starter1, ender1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter1.record()
        # ~~~~~~~~~~~~~~~
        bch, Len_in, n_heads, D_ = value.shape
        _, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        # ~~~~~~~~~~~~~~~
        torch.cuda.synchronize()
        ender1.record()
        end1 = time.time()


        start2 = time.time()
        starter2, ender2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter2.record()
        # ~~~~~~~~~~~~~~~
        for lid_, (H_, W_) in enumerate(value_spatial_shapes):
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(bch*n_heads, D_, H_, W_)
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        # ~~~~~~~~~~~~~~~
        torch.cuda.synchronize()
        ender2.record()
        end2 = time.time()



        start3 = time.time()
        starter3, ender3 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter3.record()
        # ~~~~~~~~~~~~~~~
        attention_weights = attention_weights.transpose(1, 2).reshape(bch*n_heads, 1, Len_q, n_levels*n_points)
        # ~~~~~~~~~~~~~~~
        torch.cuda.synchronize()
        ender3.record()
        end3 = time.time()



        start4 = time.time()
        starter4, ender4 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter4.record()
        # ~~~~~~~~~~~~~~~
        output = torch.stack(sampling_value_list, dim=-2).flatten(-2)
        # ~~~~~~~~~~~~~~~
        torch.cuda.synchronize()
        ender4.record()
        end4 = time.time()      



        start5 = time.time()
        starter5, ender5 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter5.record()
        # ~~~~~~~~~~~~~~~
        output = output * attention_weights
        # ~~~~~~~~~~~~~~~
        torch.cuda.synchronize()
        ender5.record()
        end5 = time.time()



        start6 = time.time()
        starter6, ender6 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter6.record()

        output = output.sum(-1).view(bch, n_heads*D_, Len_q)

        torch.cuda.synchronize()
        ender6.record()
        end6 = time.time()


        # print('Running time 1 : %s Seconds'%(end1-start1))
        # print('Running time 2 : %s Seconds'%(end2-start2))
        # print('Running time 3 : %s Seconds'%(end3-start3))
        # print('Running time 4 : %s Seconds'%(end4-start4))
        # print('Running time 5 : %s Seconds'%(end5-start5))
        # print('Running time 6 : %s Seconds'%(end6-start6))

        print('Running time 1 : %s ms'%(starter1.elapsed_time(ender1)))
        print('Running time 2 : %s ms'%(starter2.elapsed_time(ender2)))
        print('Running time 3 : %s ms'%(starter3.elapsed_time(ender3)))
        print('Running time 4 : %s ms'%(starter4.elapsed_time(ender4)))
        print('Running time 5 : %s ms'%(starter5.elapsed_time(ender5)))
        print('Running time 6 : %s ms'%(starter6.elapsed_time(ender6)))

        return output.transpose(1, 2).contiguous()







def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    # (N, Len_in, n_heads, d_model / n_heads)     Len_in = H * W * n_level = len_q
    bch, Len_in, n_heads, D_ = value.shape
    # (N, Len_q, n_heads, n_levels, n_points, 2)
    _, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
    # 分割得到特征层的value  是一个list, list的每个对应元素i, dim=1的维度大小为 H_i * W_i 
    # (n_level, N, H*W, n_heads, d_model / n_heads)
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    # 由于以下使用了 F.grid_sample(), 要求采样位置的坐标归一化到[-1, 1]范围 ((-1, -1)代表左上角; (1, 1)代表右下角)
    # 因此, 这里是将[0, 1]映射到[-1, 1]
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # bch, H_*W_, n_heads, D_    ->    bch, H_*W_, n_heads*D_    ->    bch, n_heads*D_, H_*W_ -> bch*n_heads, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(bch*n_heads, D_, H_, W_)
        # bch, Len_q, n_heads, n_points, 2    ->    bch, n_heads, Len_q, n_points, 2    ->    bch*n_heads, Len_q, n_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # 根据采样点坐标在value中插值出对应的特征
        # ps: grid_sample() 用于在归一化后的坐标对特征图进行采样 input是输入特征图，grid是归一化后的采样点，mode是插值方法
        # rid_sample(input, grid, mode='bilinear', padding_mode='border', align_corners=True)
        # 这里value_l充当被插值的特征图, 是input, 维度需要是4D/5D   (N, C, H, W)
        # sampling_grid_l_则代表采样的位置, 是grid, 最后一维2对应input中的坐标(可能是小数)  (N, H, W, 2)
        # 倒数第二、三维代表采样后输出特征图的宽、高
        # input和grid的第一个维度必须一致，最终输出的通道数与input一致，是不变的
        # bch*n_heads, D_, Len_q, n_points
        '''
        F.grid_sample 
            input(Tensor) -> 4D (N, C, H_in, W_in)    or 5D (N, C, D_in, H_in, W_in)
            grid(Tensor)  -> 4D (N, H_out, W_out, 2)  or 5D (N, D_out, H_out, W_out, 3)
            output(Tensor)-> 4D (N, C, H_out, W_out)
        '''
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    # (bch, Len_q, n_heads, n_levels, n_points)    ->    (bch, n_heads, Len_q, n_levels, n_points)    ->    (bch*n_heads, 1, Len_q, n_levels*n_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bch*n_heads, 1, Len_q, n_levels*n_points)
    # 将注意力权重和采样特征进行weighted sum
    # (bch*n_heads, D_, Len_q, n_points)    ->    (bch*n_heads, D_, Len_q, n_level, n_points)    ->    (bch*n_heads, D_, Len_q, n_level * n_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bch, n_heads*D_, Len_q)

    # (bch, Len_q, hidden_dim)
    return output.transpose(1, 2).contiguous()