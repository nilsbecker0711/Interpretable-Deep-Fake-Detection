import torch.nn.functional as F
from torch import nn
import numpy as np

class NormedDepthwiseConv2d(nn.Conv2d):
    """
    Depthwise convolution with unit norm weights.
    """
    def forward(self, in_tensor):
        shape = self.weight.shape
        w = self.weight.view(shape[0], -1)
        w = w / (w.norm(p=2, dim=1, keepdim=True) + 1e-6)
        return F.conv2d(in_tensor, w.view(shape),
                        self.bias, self.stride, self.padding, self.dilation, self.groups)

class BcosSeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, max_out=2, b=2,
                 scale=None, scale_fact=100, **kwargs):
        super().__init__()

        self.depthwise = NormedDepthwiseConv2d(in_channels, in_channels, kernel_size,
                                               stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels * max_out, 1, bias=False)
        self.out_channels = out_channels * max_out
        self.b = b
        self.max_out = max_out
        self.kernel_size = kernel_size
        self.kssq = kernel_size**2 if not isinstance(kernel_size, tuple) else np.prod(kernel_size)
        self.padding = padding
        self.detach = False

        if scale is None:
            ks_scale = kernel_size if not isinstance(kernel_size, tuple) else np.sqrt(np.prod(kernel_size))
            self.scale = (ks_scale * np.sqrt(in_channels)) / scale_fact
        else:
            self.scale = scale

    def forward(self, in_tensor):
        if self.b == 2:
            return self.fwd_2(in_tensor)
        return self.fwd_b(in_tensor)

    def explanation_mode(self, detach=True):
        self.detach = detach

    def fwd_b(self, in_tensor):
        out = self.depthwise(in_tensor)
        out = self.pointwise(out)

        if self.max_out > 1:
            bs, _, h, w = out.shape
            out = out.view(bs, -1, self.max_out, h, w)
            out = out.max(dim=2, keepdim=False)[0]

        if self.b == 1:
            return out / self.scale

        norm = (F.avg_pool2d((in_tensor ** 2).sum(1, keepdim=True), self.kernel_size, padding=self.padding,
                             stride=self.stride) * self.kssq + 1e-6).sqrt_()
        
        abs_cos = (out / norm).abs() + 1e-6

        if self.detach:
            abs_cos = abs_cos.detach()

        out = out * abs_cos.pow(self.b - 1)
        return out / self.scale

    def fwd_2(self, in_tensor):
        out = self.depthwise(in_tensor)
        out = self.pointwise(out)

        if self.max_out > 1:
            bs, _, h, w = out.shape
            out = out.view(bs, -1, self.max_out, h, w)
            out = out.max(dim=2, keepdim=False)[0]

        norm = (F.avg_pool2d((in_tensor ** 2).sum(1, keepdim=True), self.kernel_size, padding=self.padding,
                             stride=self.stride) * self.kssq + 1e-6).sqrt_()

        if self.detach:
            out = (out * out.abs().detach())
            norm = norm.detach()
        else:
            out = (out * out.abs())

        return out / (norm * self.scale)
