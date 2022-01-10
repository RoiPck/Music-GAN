from torch import nn
import torch.nn.functional as F
from custom_layers import NormalizationLayer, MusicInitFormatLayer, EqualizedConv2d
from utils import num_flat_features


class Generator(nn.Module):

    def __init__(self,
                 dimLatent,
                 depthScale0,
                 scaleSizes,
                 initBiasToZero=True,
                 leakyReluLeak=0.02,
                 normalization=True,
                 generationActivation=None,
                 dimOutput=1,
                 equalizedlR=True,
                 sizeScale0=16,
                 outputSize=0,
                 transposed=False,
                 nScales=1,
                 formatLayerType='music'):
        super(Generator, self).__init__()

        self.formatLayerType = formatLayerType

        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero

        self.sizeScale0 = sizeScale0
        self.outputSize = outputSize
        self.nScales = nScales
        self.transposed = transposed

        print()
        print(f"Size scale 0: {self.sizeScale0}")
        print()

        # Initialize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.toGreyLayers = nn.ModuleList()

        # Leaky ReLu activation
        self.leakyReLu = nn.LeakyReLU(leakyReluLeak)
        # Convolution kernels
        self.kernelSize = 3
        self.padding = 1
        # Normalization
        self.normalisationLayer = None
        if normalization:
            self.normalisationLayer = NormalizationLayer()

        # Initialize scale 0
        self.dimOutput = dimOutput
        self.initFormatLayer(dimLatent)
        self.initScale0Layer()

        # Initialize the upscaling parameters
        # Alpha is used to soften the transition when adding a new scale
        self.alpha = 0

        # Last layer activation function
        self.generationActivation = generationActivation

        self.depthScale0 = depthScale0
        self.scaleSizes = scaleSizes

    def initFormatLayer(self, dimLatentVector):
        self.dimLatent = dimLatentVector
        if self.formatLayerType == 'music':
            self.formatLayer = MusicInitFormatLayer(
                dimLatent=self.dimLatent,
                scaleDepth=self.sclaesDepth[0],
                outputShape=self.sizeScale0,
                equalizedlR=self.equalizedlR,
                initBiasToZero=self.initBiasToZero
            )

    def initScale0Layer(self):
        self.groupScale0 = nn.ModuleList()

        self.groupScale0.append(EqualizedConv2d(self.scalesDepth[0],
                                                self.scalesDepth[0],
                                                self.kernelSize,
                                                equalized=self.equalizedlR,
                                                transposed=self.transposed,
                                                initBiasToZero=self.initBiasToZero,
                                                padding=1))
        self.toGreyLayers.append(EqualizedConv2d(self.scalesDepth[0],
                                                 self.dimOutput,
                                                 1,
                                                 transposed=self.transposed,
                                                 equalized=self.equalizedlR,
                                                 initBiasToZero=self.initBiasToZero))

    # Get the size of the generated spectrogram
    def getOuputSize(self):
        if type(self.sizeScale0) == tuple:
            height = int(self.sizeScale0[0] * (2 ** (len(self.toGreyLayers))))
            width = int(self.sizeScale0[1] * (2 ** (len(self.toGreyLayers))))
            return (height, width)
        else:
            side = int(self.sizeScale0 * (2 ** (len(self.toGreyLayers))))
            return (side, side)

    # Add a new scale to the model, increase resolution by a factor 2
    def addScale(self, depthNewScale):
        if type(depthNewScale) is list:
            depthNewScale = depthNewScale[0]
        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)

        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(depthLastScale,
                                    depthNewScale,
                                    self.kernelSize,
                                    padding=self.padding,
                                    equalized=self.equalizedlR,
                                    transposed=self.transposed,
                                    initBiasToZero=self.initBiasToZero)
        self.scaleLayers[-1].append(depthNewScale,
                                    depthNewScale,
                                    self.kernelSize,
                                    padding=self.padding,
                                    equalized=self.equalizedlR,
                                    transposed=self.transposed,
                                    initBiasToZero=self.initBiasToZero)

        self.toGreyLayers[-1].append(depthNewScale,
                                    self.dimOutput,
                                    1,
                                    equalized=self.equalizedlR,
                                    transposed=self.transposed,
                                    initBiasToZero=self.initBiasToZero)

    def setNewAlpha(self, alpha):
        if alpha < 0 or alpha > 1 :
            raise ValueError("alpha must be in [0, 1]")
        if not self.toGreyLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0 is defined")
        self.alpha = alpha

    def upscale(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')

    def tensor_view(self, x):
        return x.view(x.size(0), -1, self.sizeScale0[0], self.sizeScale0[1])

    def forward(self, x, test_all_scales=False):
        output=[]
        if self.normalisationLayer is not None:
            x = self.normalisationLayer(x)

        if self.dimOutput > 1:
            x = x.view(-1, num_flat_features(x))
        # Going through format Layer
        x = self.leakyReLu(self.formatLayer(x))
        # We make sure x have the right shape
        x = self.tensor_view(x)

        x = self.normalisationLayer(x)

        #----------------------------------  Scale 0  ----------------------------------#
        # Scale 0 has no upsampling
        for convLayer in self.groupScale0:
            x = self.leakyReLu(convLayer(x))
            if self.normalisationLayer is not None:
                x = self.normalizationLayer(x)

        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.toGreyLayers[-2](x)
            y = self.upscale(y, self.scaleSizes[1])
        # Add Scale 0 for multiple scale output
        if test_all_scales:
            output.append(self.toGreyLayers[0](x))

        # ----------------------------------  Upper Scales  ----------------------------------#
        scale = 0
        for scale, layerGroup in enumerate(self.scaleLayers, 0):
            # CHECK FOR OUTPUSIZE
            x = self.upscale(x, self.scaleSizes[scale + 1])
            for convLayer in layerGroup:
                x = self.leakyReLu(convLayer(x))
                if self.normalisationLayer is not None:
                    x = self.normalisationLayer(x)
            # Add intermediate scales for multi-scale output
            if test_all_scales and scale <= len(self.scaleLayers) - 2:
                output.append(self.toGreyLayers[scale+1](x))
            if self.aklpha > 0 and scale == len(self.scaleLayers) - 2:
                y = self.toGreyLayers[-2](x)
                y = self.upscale(y, self.scaleSizes[scale + 2])

        # To Grey layer, there is no alpha parameter for now
        x = self.toGreyLayers[-1](x)
        # Blending with the lower
        if self.alpha > 0:
            s = self.alpha * y + (1.0-self.alpha) * x

        if self.generationActivation is not None:
            x = self.generationActivation(x)

        # Add last sclae to multi-scale output
        if test_all_scales and scale != 0:
            output.append(x)

        if test_all_scales:
            return output
        else:
            return x
