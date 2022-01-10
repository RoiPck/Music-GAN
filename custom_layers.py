from torch import nn
import torch
from numpy import prod, sqrt
import math

def getLayerNormalizationFactor(x):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = prod(size[1:])
    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module()):

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initBiasToZero=True):
        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forwaard(self, x, *argts):
        x = self.module(x, *args)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 bias=True,
                 transposed=False,
                 groups=1,
                 **kwargs):
        if transposed:
            ConstrainedLayer.__init__(self,
                                      nn.ConvTranspose2d(nChannelsPrevious, nChannels, kernelSize,
                                                         padding=padding, bias=bias, groups=groups),
                                      **kwargs)
        else:
            ConstrainedLayer.__init__(self,
                                      nn.Conv2d(nChannelsPrevious, nChannels, kernelSize,
                                                padding=padding, bias=bias, groups=groups),
                                      **kwargs)


class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + epsilon)


# This module aim to normalize tensor for audio
class AudioNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


# This module format the input
class MusicInitFormatLayer(nn.Module):

    def __init__(self,
                 dimLatent,
                 scaleDepth,
                 outputShape,
                 equalizedlR,
                 initBiasToZero,
                 pixelNorm=True):
        super().__init__()
        self.module = nn.ModuleList()
        self.module.extend([
            nn.ZeroPad2d((
                outputShape[1] - 1,
                outputShape[1] - 1,
                outputShape[0] - 1,
                outputShape[0] - 1)),
            EqualizedConv2d(
                dimLatent,
                scaleDepth,
                outputShape,
                equalized=equalizedlR,
                initBiasToZero=initBiasToZero,
                padding=(0,0)),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(
                scaleDepth,
                scaleDepth,
                3,
                equalized=equalizedlR,
                initBiasToZero=initBiasToZero,
                padding=1),
            nn.LeakyReLU(0.2)
        ])
        if pixelNorm:
            self.module.insert(3, AudioNorm())
            self.module.insert(6, AudioNorm())

    def forward(self, x):
        # We flatten the input before
        x = x.unsqueeze(-1).unsqueeze(-1)
        for m in self.module:
            x = m(x)
        return x