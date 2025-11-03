# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Common modules."""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.amp as amp
import torch.nn as nn
from PIL import Image

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os

    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Applies a convolution, batch normalization, and activation function to an input tensor in a neural network."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Implements a depth-wise convolution layer with optional activation for efficient spatial filtering."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """A depth-wise transpose convolutional layer for upsampling in neural networks, particularly in YOLOv5 models."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    """Transformer layer with multihead attention and linear layers, optimized by removing LayerNorm."""

    def __init__(self, c, num_heads):
        """
        Initializes a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.

        See  as described in https://arxiv.org/abs/2010.11929.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Performs forward pass using MultiheadAttention and two linear transformations with residual connections."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """A Transformer block for vision tasks with convolution, position embeddings, and Transformer layers."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block for vision tasks, adapting dimensions if necessary and stacking specified
        layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Processes input through an optional convolution, followed by Transformer layers and position embeddings for
        object detection.
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    """A bottleneck layer with optional shortcut and group convolution for efficient feature extraction."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP bottleneck layer for feature extraction with cross-stage partial connections and optional shortcuts."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    """Implements a cross convolution layer with downsampling, expansion, and optional shortcut."""

    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """Implements a CSP Bottleneck module with three convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """Extends the C3 module with cross-convolutions for enhanced feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    """C3 module with TransformerBlock for enhanced feature extraction in object detection models."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with TransformerBlock for enhanced feature extraction, accepts channel sizes, shortcut
        config, group, and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    """Extends the C3 module with an SPP layer for enhanced spatial feature extraction and customizable channels."""

    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    """Implements a C3 module with Ghost Bottlenecks for efficient feature extraction in YOLOv5."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    """Implements Spatial Pyramid Pooling (SPP) for feature extraction, ref: https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with Spatial Pyramid Pooling, ref: https://arxiv.org/abs/1406.4729, args: c1 (input channels), c2 (output channels), k (kernel sizes)."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results, and returns output
        tensor.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Implements a fast Spatial Pyramid Pooling (SPPF) layer for efficient feature extraction in YOLOv5 models."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    """Focuses spatial information into channel space using slicing and convolution for efficient feature extraction."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to concentrate width-height info into channel space with configurable convolution
        parameters.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Processes input through Focus mechanism, reshaping (b,c,w,h) to (b,4c,w/2,h/2) then applies convolution."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Implements Ghost Convolution for efficient feature extraction, see https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes GhostConv with in/out channels, kernel size, stride, groups, and activation; halves out channels
        for efficiency.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, p, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, act=act)

    def forward(self, x):
        """Performs forward pass, concatenating outputs of two convolutions on input `x`: shape (B,C,H,W)."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Efficient bottleneck layer using Ghost Convolutions, see https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see https://github.com/huawei-noah/ghostnet."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False),  # dw - now always used
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Processes input through conv and shortcut layers, returning their summed output."""
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    """Contracts spatial dimensions into channel dimensions for efficient processing in neural networks."""

    def __init__(self, gain=2):
        """Initializes a layer to contract spatial dimensions (width-height) into channels, e.g., input shape
        (1,64,80,80) to (1,256,40,40).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor to expand channel dimensions by contracting spatial dimensions, yielding output shape
        `(b, c*s*s, h//s, w//s)`.
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    """Expands spatial dimensions by redistributing channels, e.g., from (1,64,80,80) to (1,16,160,160)."""

    def __init__(self, gain=2):
        """
        Initializes the Expand module to increase spatial dimensions by redistributing channels, with an optional gain
        factor.

        Example: x(1,64,80,80) to x(1,16,160,160).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor x to expand spatial dimensions by redistributing channels, requiring C / gain^2 ==
        0.
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    """Concatenates tensors along a specified dimension for efficient tensor manipulation in neural networks."""

    def __init__(self, dimension=1):
        """Initializes a Concat module to concatenate tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
        int.
        """
        return torch.cat(x, self.d)


class WeightedFeatureFusion(nn.Module):
    """
    加权特征融合模块 - 创新亮点三：构建"加权"的特征融合颈部网络.

    核心思想：YOLOv5的FPN+PAN结构在融合不同层级的特征图时，用的是简单粗暴的Concat（拼接）。
    但不同层级的特征对于最终检测的贡献度不一定相等。我们可以让网络自己去学习不同特征图的"重要性"，
    并据此进行加权融合。

    借鉴了BiFPN（加权双向特征金字塔网络）的核心思想，引入了一个高效的加权融合机制。
    该机制让网络在训练中自主学习每个输入特征的权重，从而实现更智能、更高效的特征融合。
    """

    def __init__(self, num_inputs, eps=1e-4):
        """
        初始化加权特征融合模块.

        Args:
            num_inputs (int): 输入特征图的数量
            eps (float): 防止除零的小数值
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.eps = eps

        # 创建一个可学习的权重参数，每个输入对应一个权重
        # 初始化为1，表示开始时所有特征图权重相等
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """
        前向传播.

        Args:
            x (list): 包含多个特征图的列表，每个特征图的尺寸必须相同 [C, H, W]

        Returns:
            torch.Tensor: 加权融合后的特征图
        """
        # 归一化权重，确保其为正且和为1，增加稳定性
        w = torch.clamp(self.weights, min=0.0)  # 使用clamp确保权重为正，避免in-place操作
        w = w / (torch.sum(w, dim=0) + self.eps)  # 加上一个很小的数防止除以0

        # 计算加权和
        # 注意：这里是逐元素相加，不是通道拼接
        out = 0
        for i in range(self.num_inputs):
            out += x[i] * w[i]

        return out


class WeightedFeatureFusionConcat(nn.Module):
    """
    加权特征融合模块（通道拼接版本）.

    这个版本在加权融合后进行通道拼接，保持与原始Concat相同的输出通道数。
    适用于需要保持通道数的场景。
    """

    def __init__(self, num_inputs, eps=1e-4):
        """
        初始化加权特征融合模块（通道拼接版本）.

        Args:
            num_inputs (int): 输入特征图的数量
            eps (float): 防止除零的小数值
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.eps = eps

        # 为每个输入创建独立的权重
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """
        前向传播（通道拼接版本）.

        Args:
            x (list): 包含多个特征图的列表

        Returns:
            torch.Tensor: 加权后拼接的特征图
        """
        # 归一化权重
        w = torch.clamp(self.weights, min=0.0)  # 使用clamp确保权重为正，避免in-place操作
        w = w / (torch.sum(w, dim=0) + self.eps)

        # 对每个特征图应用权重，然后拼接
        weighted_features = []
        for i in range(self.num_inputs):
            weighted_features.append(x[i] * w[i])

        # 通道拼接
        return torch.cat(weighted_features, dim=1)


class DetectMultiBackend(nn.Module):
    """YOLOv5 MultiBackend class for inference on various backends including PyTorch, ONNX, TensorRT, and more."""

    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """Initializes DetectMultiBackend with support for various inference backends, including PyTorch and ONNX."""
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlpackage
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            core = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_profile_shape(name, 0)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wraps a TensorFlow GraphDef for inference, returning a pruned function."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                """Generates a sorted list of graph outputs excluding NoOp nodes and inputs, formatted as '<name>:0'."""
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")
        # PaddlePaddle
        elif paddle:
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle>=3.0.0")
            import paddle.inference as pdi

            w = Path(w)
            if w.is_dir():
                model_file = next(w.rglob("*.json"), None)
                params_file = next(w.rglob("*.pdiparams"), None)
            elif w.suffix == ".pdiparams":
                model_file = w.with_name("model.json")
                params_file = w
            else:
                raise ValueError(f"Invalid model path {w}. Provide model directory or a .pdiparams file.")

            if not (model_file and params_file and model_file.is_file() and params_file.is_file()):
                raise FileNotFoundError(f"Model files not found in {w}. Both .json and .pdiparams files are required.")

            config = pdi.Config(str(model_file), str(params_file))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()

        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")
            check_requirements("tritonclient[all]")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """Performs YOLOv5 inference on input images with options for augmentation and visualization."""
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            if len(y) == 2 and len(y[1].shape) != 4:
                y = list(reversed(y))
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """Converts a NumPy array to a torch tensor, maintaining device compatibility."""
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Performs a single inference warmup to initialize model weights, accepting an `imgsz` tuple for image size."""
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determines model type from file path or URL, supporting various export formats.

        Example: path='path/to/model.onnx' -> type=onnx
        """
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """Loads metadata from a YAML file, returning strides and names if the file exists, otherwise `None`."""
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    """AutoShape class for robust YOLOv5 inference with preprocessing, NMS, and support for various input formats."""

    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        """Initializes YOLOv5 model for inference, setting up attributes and preparing model for evaluation."""
        super().__init__()
        if verbose:
            LOGGER.info("Adding AutoShape... ")
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        """
        Applies to(), cpu(), cuda(), half() etc.

        to model tensors excluding parameters or registered buffers.
        """
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        """
        Performs inference on inputs with optional augment & profiling.

        Supports various formats including file, URI, OpenCV, PIL, numpy, torch.
        """
        # For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != "cpu")  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast("cuda", enabled=autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f"image{i}"  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
                files.append(Path(f).with_suffix(".jpg").name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast("cuda", enabled=autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(
                    y if self.dmb else y[0],
                    self.conf,
                    self.iou,
                    self.classes,
                    self.agnostic,
                    self.multi_label,
                    max_det=self.max_det,
                )  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    """Manages YOLOv5 detection results with methods for visualization, saving, cropping, and exporting detections."""

    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        """Initializes the YOLOv5 Detections class with image info, predictions, filenames, timing and normalization."""
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
        """Executes model predictions, displaying and/or saving outputs with optional crops and labels."""
        s, crops = "", []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(", ")
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                            crops.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "cls": cls,
                                    "label": label,
                                    "im": save_one_box(box, im, file=file, save=save),
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label if labels else "", color=colors(cls))
                    im = annotator.im
            else:
                s += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display

                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip("\n")
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops

    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        """
        Displays detection results with optional labels.

        Usage: show(labels=True)
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Saves detection results with optional labels to a specified directory.

        Usage: save(labels=True, save_dir='runs/detect/exp', exist_ok=False)
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crops detection results, optionally saves them to a directory.

        Args: save (bool), save_dir (str), exist_ok (bool).
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        """Renders detection results with optional labels on images; args: labels (bool) indicating label inclusion."""
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """
        Returns detections as pandas DataFrames for various box formats (xyxy, xyxyn, xywh, xywhn).

        Example: print(results.pandas().xyxy[0]).
        """
        new = copy(self)  # return copy
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        """
        Converts a Detections object into a list of individual detection results for iteration.

        Example: for result in results.tolist():
        """
        r = range(self.n)  # iterable
        return [
            Detections(
                [self.ims[i]],
                [self.pred[i]],
                [self.files[i]],
                self.times,
                self.names,
                self.s,
            )
            for i in r
        ]

    def print(self):
        """Logs the string representation of the current object's state via the LOGGER."""
        LOGGER.info(self.__str__())

    def __len__(self):
        """Returns the number of results stored, overrides the default len(results)."""
        return self.n

    def __str__(self):
        """Returns a string representation of the model's results, suitable for printing, overrides default
        print(results).
        """
        return self._run(pprint=True)  # print results

    def __repr__(self):
        """Returns a string representation of the YOLOv5 object, including its class and formatted results."""
        return f"YOLOv5 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    """YOLOv5 mask Proto module for segmentation models, performing convolutions and upsampling on input tensors."""

    def __init__(self, c1, c_=256, c2=32):
        """Initializes YOLOv5 Proto module for segmentation with input, proto, and mask channels configuration."""
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass using convolutional layers and upsampling on input tensor `x`."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    """YOLOv5 classification head with convolution, pooling, and dropout layers for channel transformation."""

    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        """Initializes YOLOv5 classification head with convolution, pooling, and dropout layers for input to output
        channel transformation.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Processes input through conv, pool, drop, and linear layers; supports list concatenation input."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


# =====================================================================================
# 以下为CA注意力机制代码
"""
我来详细解释CA注意力机制中每一层的输入维度变化和shape参数的含义。

## Shape的4个参数含义：
在PyTorch中，`x.size()` 或 `x.shape` 返回的是 **[N, C, H, W]**：
- **N (Batch Size)**: 批次大小，一次处理多少张图片
- **C (Channels)**: 通道数，特征图的深度
- **H (Height)**: 高度，特征图的垂直尺寸
- **W (Width)**: 宽度，特征图的水平尺寸

## CA注意力机制中每一层的维度变化：

假设输入参数：
- inp=256, oup=256, reduction=32
- 输入x的维度：**[2, 256, 32, 32]** （2张图片，256通道，32×32尺寸）

### 详细的维度变化过程：

1. **输入层**
   - `x`: **[2, 256, 32, 32]**
   - `identity = x`: **[2, 256, 32, 32]**

2. **池化操作**
   - `x_h = self.pool_h(x)`: **[2, 256, 32, 1]** (水平方向池化，保持高度)
   - `x_w = self.pool_w(x)`: **[2, 256, 1, 32]** (垂直方向池化，保持宽度)
   - `x_w.permute(0, 1, 3, 2)`: **[2, 256, 32, 1]** (转置为便于拼接)

3. **特征拼接**
   - `y = torch.cat([x_h, x_w], dim=2)`: **[2, 256, 64, 1]** (32+32=64)

4. **降维处理**
   - `mip = max(8, 256//32) = 8`
   - `y = self.conv1(y)`: **[2, 8, 64, 1]** (256→8通道)
   - `y = self.bn1(y)`: **[2, 8, 64, 1]** (批归一化，维度不变)
   - `y = self.act(y)`: **[2, 8, 64, 1]** (激活函数，维度不变)

5. **特征分割**
   - `x_h, x_w = torch.split(y, [32, 32], dim=2)`:
     - `x_h`: **[2, 8, 32, 1]**
     - `x_w`: **[2, 8, 32, 1]**
   - `x_w = x_w.permute(0, 1, 3, 2)`: **[2, 8, 1, 32]**

6. **生成注意力权重**
   - `a_h = self.conv_h(x_h).sigmoid()`: **[2, 256, 32, 1]** (8→256通道)
   - `a_w = self.conv_w(x_w).sigmoid()`: **[2, 256, 1, 32]** (8→256通道)

7. **最终输出**
   - `out = identity * a_w * a_h`: **[2, 256, 32, 32]**
   - 通过广播机制：
     - identity: [2, 256, 32, 32]
     - a_w: [2, 256, 1, 32] → 广播到 [2, 256, 32, 32]
     - a_h: [2, 256, 32, 1] → 广播到 [2, 256, 32, 32]

## 关键维度变化总结：

| 步骤 | 操作 | 输入维度 | 输出维度 |
|------|------|----------|----------|
| 1 | 原始输入 | - | [N, C, H, W] |
| 2 | 水平池化 | [N, C, H, W] | [N, C, H, 1] |
| 3 | 垂直池化+转置 | [N, C, H, W] | [N, C, W, 1] |
| 4 | 特征拼接 | 两个[N, C, *, 1] | [N, C, H+W, 1] |
| 5 | 降维卷积 | [N, C, H+W, 1] | [N, mip, H+W, 1] |
| 6 | 特征分割 | [N, mip, H+W, 1] | [N, mip, H, 1] + [N, mip, 1, W] |
| 7 | 生成权重 | [N, mip, H, 1] | [N, oup, H, 1] |
| 8 | 生成权重 | [N, mip, 1, W] | [N, oup, 1, W] |
| 9 | 最终输出 | [N, oup, H, W] + 权重 | [N, oup, H, W] |

这样的设计能够捕获空间位置信息并将其编码到注意力权重中，从而增强特征表示能力。
"""
# =====================================================================================


class h_swish(nn.Module):
    """Hard Swish activation function for improved efficiency."""

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        """Applies Hard Swish activation function."""
        return x * torch.nn.functional.relu6(x + 3.0, inplace=self.inplace) / 6.0


class CoordAtt(nn.Module):
    """Coordinate Attention mechanism for enhanced feature representation."""

    def __init__(self, inp, oup, reduction=32):
        """Initialize Coordinate Attention with input channels, output channels and reduction ratio."""
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """Forward pass applying coordinate attention mechanism."""
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# =====================================================================================
# CA注意力机制代码结束
# =====================================================================================


# =====================================================================================
# 以下为SE注意力机制代码
"""
SE注意力机制（Squeeze-and-Excitation）实现

SE注意力机制是一种通道注意力机制，通过学习通道间的相互依赖关系来重新校准特征响应。
主要包含两个操作：
1. Squeeze：通过全局平均池化将每个通道的空间特征压缩为一个标量
2. Excitation：通过两个全连接层学习通道间的非线性交互，生成通道权重

论文：Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)

SE模块的优势：
- 轻量级：参数量很少，计算开销小
- 即插即用：可以轻松集成到现有网络架构中
- 性能提升：能够显著提升模型的表征能力
- 通道注意力：专注于"什么"特征是重要的

实现细节：
- 使用全局平均池化进行特征压缩
- 使用两个FC层构建激励操作，中间使用ReLU激活
- 最后使用Sigmoid生成0-1之间的通道权重
- 通过逐元素相乘将权重应用到原始特征上
"""


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.

    SE注意力机制模块，通过学习通道间的相互依赖关系来重新校准特征响应。
    支持自适应通道数检测，可以在YAML配置中不指定通道数。

    Args:
        c1 (int, optional): 输入通道数，如果为None则在第一次前向传播时自动检测
        reduction (int): 降维比例，默认为16
    """

    def __init__(self, c1=None, reduction=16):
        """初始化SE模块."""
        super().__init__()
        self.c1 = c1
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

        # 如果提供了通道数，立即初始化FC层
        if c1 is not None:
            self._build_fc_layers(c1)
        else:
            self.fc = None  # 延迟初始化

    def _build_fc_layers(self, c1):
        """构建FC层."""
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // self.reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),  # 激活
            nn.Linear(c1 // self.reduction, c1, bias=False),  # 升维
            nn.Sigmoid(),  # 生成权重
        )

    def forward(self, x):
        """前向传播."""
        b, c, _, _ = x.size()

        # 如果FC层未初始化，则根据输入自动初始化
        if self.fc is None:
            self.c1 = c
            self._build_fc_layers(c)
            # 将FC层移动到正确的设备
            if x.is_cuda:
                self.fc = self.fc.cuda()

        # Squeeze: 全局平均池化
        y = self.avg_pool(x).view(b, c)
        # Excitation: 通过FC层生成通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # 将权重应用到原始特征
        return x * y.expand_as(x)


class SEConv(nn.Module):
    """
    SE-Conv: 集成SE注意力机制的卷积模块.

    在标准卷积后添加SE注意力机制，增强特征表示能力。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, reduction=16):
        """初始化SE-Conv模块."""
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)  # 标准卷积
        self.se = SEBlock(c2, reduction)  # SE注意力机制

    def forward(self, x):
        """前向传播."""
        x = self.conv(x)
        x = self.se(x)
        return x


class SEBottleneck(nn.Module):
    """
    SE-Bottleneck: 集成SE注意力机制的瓶颈模块.

    在标准Bottleneck的基础上添加SE注意力机制。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, reduction=16):
        """初始化SE-Bottleneck模块."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.se = SEBlock(c2, reduction)  # SE注意力机制
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """前向传播."""
        if self.add:
            return x + self.se(self.cv2(self.cv1(x)))
        else:
            return self.se(self.cv2(self.cv1(x)))


class C3SE(C3):
    """
    C3-SE: 集成SE注意力机制的C3模块.

    将C3模块中的Bottleneck替换为SEBottleneck，增强特征表示能力。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, reduction=16):
        """初始化C3-SE模块."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(SEBottleneck(c_, c_, shortcut, g, e=1.0, reduction=reduction) for _ in range(n)))


# =====================================================================================
# SE注意力机制代码结束
# =====================================================================================


# =====================================================================================
# 以下为MoE（Mixture of Experts）架构代码
"""
MoE（Mixture of Experts）架构实现

MoE是一种稀疏激活的神经网络架构，通过使用多个专家网络和门控机制来提升模型容量，
同时保持计算效率。主要组件包括：

1. Expert Networks: 多个专家网络，每个专家专门处理特定类型的输入
2. Gating Network: 门控网络，决定哪些专家被激活以及激活权重
3. Top-K Selection: 只激活Top-K个专家，实现稀疏计算
4. Load Balancing: 负载均衡机制，确保专家使用均匀

论文参考：
- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- "Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)

MoE的优势：
- 模型容量大：可以有很多专家，但每次只激活少数几个
- 计算效率高：稀疏激活，实际计算量不会线性增长
- 专业化：不同专家可以学习处理不同类型的特征
- 可扩展性：容易扩展到更多专家

适用场景：
- 大规模数据集
- 多样化的输入模式
- 需要高容量但保持效率的场景
"""


import torch.nn.functional as F


class Expert(nn.Module):
    """
    单个专家网络.

    每个专家是一个简单的前馈网络，通常包含两个线性层和激活函数。
    在目标检测中，我们将其适配为卷积专家。
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, expert_type="conv"):
        """
        初始化专家网络.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            p: 填充
            g: 分组
            act: 是否使用激活函数
            expert_type: 专家类型 ('conv', 'bottleneck', 'dwconv')
        """
        super().__init__()
        self.expert_type = expert_type

        if expert_type == "conv":
            # 标准卷积专家
            self.expert = Conv(c1, c2, k, s, p, g, act=act)
        elif expert_type == "bottleneck":
            # 瓶颈结构专家
            c_ = c2 // 4
            self.expert = nn.Sequential(
                Conv(c1, c_, 1, 1, act=act), Conv(c_, c_, k, s, p, g, act=act), Conv(c_, c2, 1, 1, act=False)
            )
        elif expert_type == "dwconv":
            # 深度可分离卷积专家
            self.expert = nn.Sequential(DWConv(c1, c1, k, s, act=act), Conv(c1, c2, 1, 1, act=act))
        else:
            raise ValueError(f"Unsupported expert_type: {expert_type}")

    def forward(self, x):
        """前向传播."""
        return self.expert(x)


class SparseGating(nn.Module):
    """
    稀疏门控网络.

    负责计算每个专家的权重，并实现Top-K选择和负载均衡。
    """

    def __init__(self, c1, num_experts, top_k=2, capacity_factor=1.25):
        """
        初始化门控网络.

        Args:
            c1: 输入通道数
            num_experts: 专家数量
            top_k: 选择的专家数量
            capacity_factor: 容量因子，用于负载均衡
        """
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)  # 确保top_k是整数
        self.capacity_factor = float(capacity_factor)

        # 门控网络：全局平均池化 + 线性层
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c1, num_experts), nn.Softmax(dim=-1))

        # 噪声用于训练时的负载均衡
        self.noise_std = 0.1

    def forward(self, x):
        """
        前向传播.

        Returns:
            gates: 门控权重 [batch_size, num_experts]
            indices: 选中的专家索引 [batch_size, top_k]
            load_balancing_loss: 负载均衡损失
        """
        x.size(0)

        # 计算门控权重
        gates = self.gate(x)  # [batch_size, num_experts]

        # 训练时添加噪声以促进负载均衡
        if self.training:
            noise = torch.randn_like(gates) * self.noise_std
            gates = gates + noise
            gates = F.softmax(gates, dim=-1)

        # Top-K选择
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)

        # 重新归一化
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)

        # 计算负载均衡损失
        load_balancing_loss = self._compute_load_balancing_loss(gates)

        return top_k_gates, top_k_indices, load_balancing_loss

    def _compute_load_balancing_loss(self, gates):
        """计算负载均衡损失."""
        # 计算每个专家的平均使用率
        expert_usage = gates.mean(dim=0)  # [num_experts]

        # 理想情况下每个专家的使用率应该是 1/num_experts
        target_usage = 1.0 / self.num_experts

        # 计算均方误差作为负载均衡损失
        load_balancing_loss = F.mse_loss(expert_usage, torch.full_like(expert_usage, target_usage))

        return load_balancing_loss


class MoELayer(nn.Module):
    """
    MoE层.

    结合多个专家网络和门控机制，实现稀疏激活的混合专家架构。
    """

    def __init__(self, c1, c2, num_experts=4, top_k=2, expert_type="conv", k=3, s=1, p=None, g=1, act=True):
        """
        初始化MoE层.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            num_experts: 专家数量
            top_k: 激活的专家数量
            expert_type: 专家类型
            k, s, p, g, act: 卷积参数
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 创建专家网络
        self.experts = nn.ModuleList([Expert(c1, c2, k, s, p, g, act, expert_type) for _ in range(num_experts)])

        # 门控网络
        self.gate = SparseGating(c1, num_experts, top_k)

        # 用于存储负载均衡损失（使用register_buffer避免deepcopy问题）
        self.register_buffer("load_balancing_loss", torch.tensor(0.0))

    def forward(self, x):
        """前向传播."""
        batch_size = x.size(0)

        # 获取门控权重和选中的专家
        top_k_gates, top_k_indices, load_balancing_loss = self.gate(x)
        self.load_balancing_loss.data = load_balancing_loss.data

        # 初始化输出
        output = torch.zeros_like(self.experts[0](x))

        # 对每个样本处理
        for i in range(batch_size):
            sample_output = torch.zeros_like(output[i : i + 1])

            # 对选中的专家进行加权求和
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j].item()
                expert_weight = top_k_gates[i, j]

                expert_output = self.experts[expert_idx](x[i : i + 1])
                sample_output += expert_weight * expert_output

            output[i : i + 1] = sample_output

        return output


class MoEBottleneck(nn.Module):
    """
    MoE瓶颈模块.

    将MoE层集成到瓶颈结构中，用于替换标准的Bottleneck。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, num_experts=4, top_k=2):
        """初始化MoE瓶颈模块."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = MoELayer(c_, c2, num_experts, top_k, expert_type="conv", k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """前向传播."""
        if self.add:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))


class C3MoE(C3):
    """
    C3-MoE模块.

    将C3模块中的Bottleneck替换为MoEBottleneck，实现混合专家架构。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, num_experts=4, top_k=2):
        """初始化C3-MoE模块."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(
            *(MoEBottleneck(c_, c_, shortcut, g, e=1.0, num_experts=num_experts, top_k=top_k) for _ in range(n))
        )

    def get_load_balancing_loss(self):
        """获取所有MoE层的负载均衡损失."""
        total_loss = 0.0
        count = 0
        for module in self.m:
            if hasattr(module, "cv2") and hasattr(module.cv2, "load_balancing_loss"):
                loss_val = module.cv2.load_balancing_loss
                if isinstance(loss_val, torch.Tensor):
                    total_loss += loss_val.item()
                else:
                    total_loss += float(loss_val)
                count += 1
        return total_loss / max(count, 1)


class MoEConv(nn.Module):
    """
    MoE卷积模块.

    直接使用MoE层作为卷积操作的替代。
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, num_experts=4, top_k=2, expert_type="conv"):
        """初始化MoE卷积模块."""
        super().__init__()
        self.moe = MoELayer(c1, c2, num_experts, top_k, expert_type, k, s, p, g, act)

    def forward(self, x):
        """前向传播."""
        return self.moe(x)

    def get_load_balancing_loss(self):
        """获取负载均衡损失."""
        loss_val = self.moe.load_balancing_loss
        if isinstance(loss_val, torch.Tensor):
            return loss_val.item()
        else:
            return float(loss_val)


class AdaptiveMoE(nn.Module):
    """
    自适应MoE模块.

    根据输入特征的复杂度动态调整激活的专家数量。
    """

    def __init__(self, c1, c2, max_experts=6, min_top_k=1, max_top_k=3, k=3, s=1, p=None, g=1, act=True):
        """
        初始化自适应MoE模块.

        Args:
            max_experts: 最大专家数量
            min_top_k: 最小激活专家数
            max_top_k: 最大激活专家数
        """
        super().__init__()
        self.max_experts = max_experts
        self.min_top_k = min_top_k
        self.max_top_k = max_top_k

        # 创建专家网络
        self.experts = nn.ModuleList([Expert(c1, c2, k, s, p, g, act, "conv") for _ in range(max_experts)])

        # 复杂度评估网络
        self.complexity_estimator = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c1, 1), nn.Sigmoid())

        # 门控网络
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c1, max_experts), nn.Softmax(dim=-1))

    def forward(self, x):
        """前向传播."""
        batch_size = x.size(0)

        # 评估输入复杂度
        complexity = self.complexity_estimator(x)  # [batch_size, 1]

        # 根据复杂度确定激活的专家数量
        top_k = self.min_top_k + (complexity * (self.max_top_k - self.min_top_k)).round().int()
        top_k = torch.clamp(top_k, self.min_top_k, self.max_top_k)

        # 计算门控权重
        gates = self.gate(x)  # [batch_size, max_experts]

        # 初始化输出
        output = torch.zeros_like(self.experts[0](x))

        # 对每个样本处理
        for i in range(batch_size):
            sample_top_k = top_k[i].item()
            sample_gates = gates[i]

            # 选择Top-K专家
            top_k_gates, top_k_indices = torch.topk(sample_gates, sample_top_k)
            top_k_gates = top_k_gates / (top_k_gates.sum() + 1e-8)

            sample_output = torch.zeros_like(output[i : i + 1])

            # 对选中的专家进行加权求和
            for j in range(sample_top_k):
                expert_idx = top_k_indices[j].item()
                expert_weight = top_k_gates[j]

                expert_output = self.experts[expert_idx](x[i : i + 1])
                sample_output += expert_weight * expert_output

            output[i : i + 1] = sample_output

        return output


# =====================================================================================
# 混合MoE架构代码开始 (基于GUIDE/MOE2.md设计)
# =====================================================================================


class SharedExpert(nn.Module):
    """
    共享专家网络.

    所有输入都必须经过的"通识"专家，提供基础特征处理能力。
    保证模型性能下限，加速收敛，降低门控网络学习负担。
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        """
        初始化共享专家.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            p: 填充
            g: 分组
            act: 是否使用激活函数
        """
        super().__init__()
        # 共享专家使用标准的Bottleneck结构
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, k, s, p, g, act=act)

    def forward(self, x):
        """前向传播."""
        return self.cv2(self.cv1(x))


class HybridMoELayer(nn.Module):
    """
    混合MoE层.

    结合共享专家和稀疏专家的混合架构：
    - 共享专家：所有输入都经过，提供基础特征
    - 专业专家：通过门控选择，提供专业化特征
    - 最终输出：共享特征 + 专业特征
    """

    def __init__(self, c1, c2, num_experts=12, top_k=2, shared_ratio=0.25, k=3, s=1, p=None, g=1, act=True):
        """
        初始化混合MoE层.

        Args:
            c1: 输入通道数
            c2: 输出通道数
            num_experts: 专业专家数量
            top_k: 激活的专业专家数量
            shared_ratio: 共享专家通道比例
            k, s, p, g, act: 卷积参数
        """
        super().__init__()
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)  # 确保top_k是整数
        self.shared_ratio = float(shared_ratio)

        # 计算通道分配
        shared_channels = int(c2 * shared_ratio)
        expert_channels = c2 - shared_channels

        # 共享专家
        self.shared_expert = SharedExpert(c1, shared_channels, k, s, p, g, act)

        # 专业专家网络
        self.experts = nn.ModuleList(
            [Expert(c1, expert_channels, k, s, p, g, act, "bottleneck") for _ in range(num_experts)]
        )

        # 门控网络
        self.gate = SparseGating(c1, num_experts, top_k)

        # 存储输出通道数，用于前向传播
        self.expert_channels = expert_channels

        # 用于存储负载均衡损失
        self.register_buffer("load_balancing_loss", torch.tensor(0.0))

    def forward(self, x):
        """前向传播."""
        batch_size = x.size(0)

        # 1. 共享专家处理（所有输入都经过）
        shared_output = self.shared_expert(x)

        # 2. 门控网络选择专业专家
        top_k_gates, top_k_indices, load_balancing_loss = self.gate(x)
        self.load_balancing_loss.data = load_balancing_loss.data

        # 3. 专业专家处理
        expert_output = torch.zeros(batch_size, self.expert_channels, x.size(2), x.size(3), device=x.device)

        # 对每个样本处理
        for i in range(batch_size):
            sample_expert_output = torch.zeros_like(expert_output[i : i + 1])

            # 对选中的专家进行加权求和
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j].item()
                expert_weight = top_k_gates[i, j]

                expert_result = self.experts[expert_idx](x[i : i + 1])
                sample_expert_output += expert_weight * expert_result

            expert_output[i : i + 1] = sample_expert_output

        # 4. 最终输出 = 共享特征 + 专业特征
        final_output = torch.cat([shared_output, expert_output], dim=1)

        return final_output


class HybridMoEBottleneck(nn.Module):
    """
    混合MoE瓶颈模块.

    将混合MoE层集成到瓶颈结构中，用于替换标准的Bottleneck。
    """

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, num_experts=12, top_k=2, shared_ratio=0.25):
        """初始化混合MoE瓶颈模块."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = HybridMoELayer(c_, c2, num_experts, top_k, shared_ratio, k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """前向传播."""
        if self.add:
            return x + self.cv2(self.cv1(x))
        else:
            return self.cv2(self.cv1(x))


class C3HybridMoE(C3):
    """
    C3-混合MoE模块.

    将C3模块中的Bottleneck替换为HybridMoEBottleneck，实现混合专家架构。
    结合了共享专家的稳定性和专业专家的专业化能力。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, num_experts=12, top_k=2, shared_ratio=0.25):
        """初始化C3-混合MoE模块."""
        # 注意：这里不能直接调用super().__init__，因为我们需要重写self.m
        # super().__init__(c1, c2, n, shortcut, g, e)

        # 手动初始化C3的基础部分
        super(C3, self).__init__()  # 调用nn.Module的初始化
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        # 创建混合MoE Bottleneck序列
        self.m = nn.Sequential(
            *(
                HybridMoEBottleneck(
                    c_, c_, shortcut, g, e=1.0, num_experts=num_experts, top_k=top_k, shared_ratio=shared_ratio
                )
                for _ in range(n)
            )
        )

    def get_load_balancing_loss(self):
        """获取所有混合MoE层的负载均衡损失."""
        total_loss = 0.0
        count = 0
        for module in self.m:
            if hasattr(module, "cv2") and hasattr(module.cv2, "load_balancing_loss"):
                loss_val = module.cv2.load_balancing_loss
                if isinstance(loss_val, torch.Tensor):
                    total_loss += loss_val.item()
                else:
                    total_loss += float(loss_val)
                count += 1
        return total_loss / max(count, 1)


# =====================================================================================
# 混合MoE架构代码结束
# =====================================================================================


# =====================================================================================
# MoE架构代码结束
# =====================================================================================
