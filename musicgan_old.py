from torch import nn
import torch
import torch.functional as F
from torchsummary import summary
import torchaudio
from torch.utils.data import DataLoader
from instrumentssounddataset import InstrumentSoundDataset
import os
import pandas as pd
from functools import partial


class MusicGAN(nn.Module):

    def __init__(self,
                 shape,
                 features, # [256, 128, 128, 128, 64, 32]
                 conv_filters=(3, 3),
                 z_dim=128):
        self.shape = shape
        self.conv_filters = conv_filters
        self.z_dim = z_dim
        self.features = features
        self.num_blocks = len(features)

        self.kernel_size = 3
        self.padding = 1

        self.model_input = None

        self.generator = None
        self.discriminator = None
        self.model = None

        self._build()

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()
        self.model.summary()

    def _build(self):
        self._build_generator()
        self._build_discriminator()
        self._build_gan()

    def _build_gan(self):
        pass

    def _build_generator(self):
        gen_input = self._add_gen_input()
        gen_conv_blocks = self._add_gen_conv_block(gen_input)

    def _add_gen_input(self):
        input_module = nn.Sequential(
            nn.ZeroPad2d((
                self.shape[1] - 1,
                self.shape[1] - 1,
                self.shape[0] - 1,
                self.shape[0] - 1
            )),
            nn.Conv2d(
                self.z_dim,
                1,
                self.shape,
                padding=(0,0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                1,
                1,
                self.kernel_size,
                padding=self.padding),
            nn.LeakyReLU(0.2))
        return input_module

    def _add_gen_blocks(self, gen_input):
        x = gen_input
        for block_index in range(self.num_blocks):
            x = self._add_gen_block(block_index, x)
        return x

    def _upscale(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')

    def _add_gen_block(self, block_index, x):
        block_number = block_index + 1
        block_module = nn.Sequential(
            nn.ConvTranspose2d(
                block_number - 1,
                block_number,
                self.kernel_size,
                padding=self.padding),
            nn.Conv2d(
                block_number,
                block_number,
                self.kernel_size,
                padding=self.padding),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                block_number,
                block_number,
                self.kernel_size,
                padding=self.padding),
            nn.LeakyReLU(0.2)
        )

    def _build_discriminator(self):
        pass





class Generator(nn.Module):

    def __init__(self,
                 dimLatent,
                 depthScale0,
                 scaleSizes,
                 leakyReLuLeak=0.2,
                 dimOutput=1,
                 sizeScale0=16,
                 outputSize=0,
                 n_scales=1,
                 generationActivation=None):
        super(Generator, self).__init__()

        # Dimensions params
        self.dimLatent = dimLatent
        self.dimOutput = dimOutput
        # Scale 0 params
        self.depthScale0 = depthScale0
        self.sizeScale0 = sizeScale0
        # Scales params
        self.n_scales = n_scales
        self.scaleSizes = scaleSizes
        self.outputSize = outputSize
        # Leaky ReLu params
        self.leakyReLuLeak = leakyReLuLeak

        # Initialize Scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.toGreyLayers = nn.ModuleList()

        # Set up leaky ReLu activation
        self.leakyReLu = torch.nn.LeakyReLU(leakyReLuLeak)

        # Hyper Parameters for convolution
        self.kernelSize = 3
        self.padding = 1

        # Initialize the first scale
        self._init_format_layer()
        self._init_scale_0_layer()

        # Initialize alpha for the training
        # When a new scale is added to the network, we smoothly merge the output with previous layer
        self.aplha = 0

        # Generation Activation function
        self.generationActivation = generationActivation

    def __init_format_layer(self):
        # Make sure to make x = x.unsqueeze(-1).unsqueeze(-1) before
        self.initFormatLayer = nn.ModuleList()
        self.initFormatLayer.extend([
            nn.ZeroPad2d((
                self.sizeScale0[1] - 1,
                self.sizeScale0[1] - 1,
                self.sizeScale0[0] - 1,
                self.sizeScale0[0] - 1
            )),
            nn.Conv2d(
                self.dimLatent,
                1,
                self.sizeScale0,
                padding=(0, 0)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                1,
                1,
                self.kernelSize,
                padding=self.padding),
            nn.LeakyReLU(0.2)
        ])

    def _init_scale_0_layer(self):
        self.groupScale0 = nn.ModuleList()

        self.groupScale0.append(nn.Conv2D(
            
        ))

        


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":

    ANNOTATIONS_FILE = "./data/prepared/sound_data.csv"
    AUDIO_DIR = "./data/prepared/sounds"
    SAMPLE_RATE = 44100
    DURATION = 5
    NUM_SAMPLE = DURATION * SAMPLE_RATE
    BATCH_SIZE = 1

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    spectro_params = {
        "n_fft": 1024,
        "hop_length": 512,
        "win_length": 2048,
        "return_complex": False
    }

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=spectro_params["n_fft"],
        hop_length=spectro_params["hop_length"],
        win_length=spectro_params["win_length"],
        center=True,
        pad_mode="reflect",
        power=None,
        return_complex=spectro_params["return_complex"]
    )

    isd = InstrumentSoundDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLE,
        device
    )

    spectrogram_to_signal = torchaudio.transforms.InverseSpectrogram(
        n_fft=spectro_params["n_fft"],
        hop_length=spectro_params["hop_length"],
        win_length=spectro_params["win_length"],
        # Using partial to modify the method default parameters so as to use the right device
        window_fn=partial(torch.hann_window, device=isd.device)
    )

    disc = Discriminator().to(device)
    gen = Generator().to(device)

    train_loader = DataLoader(isd, batch_size=BATCH_SIZE)
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)

        out = disc(input)
        print(out)
        break
    # summary