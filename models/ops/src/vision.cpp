/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

/*
使用pybind11库的C++代码，用于在Python中绑定两个函数，ms_deform_attn_forward 和 ms_deform_attn_backward
PYBIND11_MODULE 是pybind11的一个宏，用于定义一个Python模块。
TORCH_EXTENSION_NAME是模块的名字，而m是一个代表模块的对象，我们可以在其上定义各种Python功能
*/


#include "ms_deform_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
