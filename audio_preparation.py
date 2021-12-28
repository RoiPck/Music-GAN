import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import io
import os
import math
import tarfile
import multiprocessing

import scipy
import librosa
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython.display import Audio, display

from IPython.display import Audio, display


if __name__ == "__main__":
    # Sample parameters
    RESAMPLE_RATE = 44100
    # Samples path parameters
    #SAMPLE_DIR = os.path.join("data", "not_prepared", "kick")
    SAMPLE_NAME = "mhak Kick 25 F"
    SAMPLE_NAME = "bass-lancet"
    # Transform parameters

    #Loading sample

    SAMPLE_WAV_PATH = os.path.join(SAMPLE_DIR, SAMPLE_NAME + ".wav")
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_PATH)

    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 64
    n_freqs = 201

    audio_transform = nn.Sequential(
        T.Resample(sample_rate, RESAMPLE_RATE),
        T.MelSpectrogram(sample_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)
    )
    mel_specgram = audio_transform(waveform)

    print("Shape of spectrogram: {}".format(mel_specgram.size()))

    height = 120
    width = 500

    #plt.figure()
    #data = mel_specgram.log2()[0, :, :].detach().numpy()
    data = mel_specgram.detach().numpy()

    fig, ax = plt.subplots()
    ax.imshow(data, origin='lower', cmap='plasma')
    ax.autoscale(False)
    ax.plot()
    #plt.xlim([0, 500])
    plt.show()


    #print(torch.all(waveform.eq(new_wave)))

