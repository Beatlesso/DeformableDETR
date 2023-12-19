/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <vector>
#include "cuda/ms_deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>


at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    // 首先输入的5个张量都得是连续的
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");

    // 此外5个张量都必需是 CUDA tensor
    AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.type().is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor");

    // 拿出对应的参数
    const int batch = value.size(0);
    // 这里 spatial_size 是所有特征图的大小之和
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    // d_model / n_heads
    const int channels = value.size(3);
    // 下面的信息都是对应的
    const int num_levels = spatial_shapes.size(0);
    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);
    // 这个比较特别，取了min
    const int im2col_step_ = std::min(batch, im2col_step);

    // batch 必需是 im2col_step_ 的倍数
    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);
    // 创建输出对应的 0 张量，options 用于指定新张量的数据类型和其他选项，这通常与 value 张量具有相同的数据类型和设备
    auto output = at::zeros({batch, num_query, num_heads, channels}, value.options());
    // 注意 batch_n 是 im2col_step_
    const int batch_n = im2col_step_;

    auto output_n = output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    // per_value_size 表示所有特征图的 元素总数：spatial_size * d_model
    auto per_value_size = spatial_size * num_heads * channels;
    // per_sample_loc_size 表示所有采样点坐标的 元素总数
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    // per_attn_weight_size 表示所有采样点注意力权重的 元素总数
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;


    // 每次处理 im2col_step_ 个数据
    for (int n = 0; n < batch/im2col_step_; ++n) // 注意这里枚举变量是 n
    {
        // tensor.select(0，index) 等价于tensor[index]，所以columns就是当前处理数据的对应output部分
        auto columns = output_n.select(0, n);
        /*
            AT_DISPATCH_FLOATING_TYPES是一个宏，用于实现动态分发机制，可以替换成AT_DISPATCH_ALL_TYPES
            即它会在运行时，根据输入具体的数值类型，去决定之前CUDA kernel模块函数需要实例化为哪种函数
            它有三个参数，第一个是tensor的数据类型，第二个是用于显示错误的信息，第三个是个匿名函数
            ([&]{ })内写cuda的__global__ kernel函数
            Pytorch拓展C++ 以及 CUDA参考：https://blog.csdn.net/hellyou/article/details/105382069


            Stream是帮助我们实现以上两个并行的重要工具。基本的概念是：
            将数据拆分称许多块，每一块交给一个Stream来处理。
            每一个Stream包含了三个步骤：1）将属于该Stream的数据从CPU内存转移到GPU内存，2）GPU进行运算并将结果保存在GPU内存，3）将该Stream的结果从GPU内存拷贝到CPU内存。
            所有的Stream被同时启动，由GPU的scheduler决定如何并行。
            CUDA Stream：https://zhuanlan.zhihu.com/p/51402722
        */
        // 总的来说，下面就是一个自动分发函数，每次取 im2col_step_ 个样例数据进行计算
        AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda", ([&] {
            // 具体来说调用了 ms_deformable_im2col_cuda 这个函数
            /*
                cuda::getCurrentCUDAStream()
                Get the current CUDA stream, for the passed CUDA device, or for the current device if no device index is passed.
                The current CUDA stream will usually be the default CUDA stream for the device, 
                but it may be different if someone called ‘setCurrentCUDAStream’ or used ‘StreamGuard’ or ‘CUDAStreamGuard’.
            */
            ms_deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),     
                // input.data<scalar_t>() 把input的数据转换成scalar_t类型并且返回一个头指针，这里猜测 scalar_t 是 value.type()
                // 该数据是一个一维的连续存储的地址，访问数据的方式和c语言指针使用方法一样
                // 每次都处理 im2col_step_，所以第n次的地址偏移就是 n * im2col_step_ * per_value_size，后面的类似
                value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data<int64_t>(),
                level_start_index.data<int64_t>(),
                sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                columns.data<scalar_t>());

        }));
    }

    output = output.view({batch, num_query, num_heads*channels});

    return output;
}


std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{

    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
    AT_ASSERTM(spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous");
    AT_ASSERTM(level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous");
    AT_ASSERTM(sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous");
    AT_ASSERTM(attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous");
    AT_ASSERTM(grad_output.is_contiguous(), "grad_output tensor has to be contiguous");

    AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(level_start_index.type().is_cuda(), "level_start_index must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor");
    AT_ASSERTM(grad_output.type().is_cuda(), "grad_output must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = grad_output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto grad_output_g = grad_output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda", ([&] {
            ms_deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                    grad_output_g.data<scalar_t>(),
                                    value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                                    spatial_shapes.data<int64_t>(),
                                    level_start_index.data<int64_t>(),
                                    sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                    batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                    grad_value.data<scalar_t>() +  n * im2col_step_ * per_value_size,
                                    grad_sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    grad_attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size);

        }));
    }

    return {
        grad_value, grad_sampling_loc, grad_attn_weight
    };
}