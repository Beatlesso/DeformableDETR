# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

# build_position_encoding 用于创建位置编码
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # 检查是否需要训练
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # 如果需要反回多尺度特征图
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        
        # IntermediateLayerGetter 获取一个Model中你指定要获取的哪些层的输出，然后这些层的输出会在一个有序的字典中，
        # 字典中的key就是刚开始初始化这个类传进去的，value就是feature经过指定需要层的输出。
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # 前向中输入的是NestedTensor这个类的实例，实质就是将图像张量与对应的mask封装到一起
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将mask插值到与输出特征图一致,  input – 输入张量,  size - 输出大小
            mask = F.interpolate(input=m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        # 使用getattr函数从torchvision.models模块中获取一个名为name的模型  默认resnet50
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation], # 表示是否传入空洞卷积参数   第一个和第二个卷积层中步长保持原样  
            pretrained=is_main_process(), norm_layer=norm_layer) # is_main_process 如果是单卡模式，那么将加载预训练的权重；否则，将不加载。
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        # 用给定参数创建BackboneBase
        super().__init__(backbone, train_backbone, return_interm_layers)
        # 如果是空洞卷积，最后的步长除以2
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


# 将backbone和position_embedding和并成Joiner
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        # Sequential类中的self[0]通常代表第一个被添加的子模块
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        # 所以最后backbone的输出就是 所需要的不同尺度特征图以及位置编码
        return out, pos


def build_backbone(args):
    # 建立位置编码
    position_embedding = build_position_encoding(args)
    # 判断要不要训练backbone
    train_backbone = args.lr_backbone > 0
    # 如果是图像分割或者多尺度
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    # 传入对应参数创建backbone
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    return model
