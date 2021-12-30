import torch
import torchaudio
from torch.utils.data import Dataset
import os
from functools import partial
import pandas as pd


class InstrumentSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _get_audio_sample_path(self, index):
        filename = self.annotations.iloc[index, 0]
        path = os.path.join("data", "prepared", "sounds", filename)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 2]

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


if __name__ == "__main__":
    ANNOTATIONS_FILE = "./data/prepared/sound_data.csv"
    AUDIO_DIR =  "./data/prepared/sounds"
    SAMPLE_RATE = 44100
    DURATION = 5
    NUM_SAMPLE = DURATION * SAMPLE_RATE

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_params = {
        "sample_rate" : SAMPLE_RATE,
        "n_fft" : 1024,
        "hop_length" : 512,
        "win_length" : 1024,
        "n_mels" : 64
    }

    #It seems to be the better parameters, for kicks at least
    spectro_params = {
        "n_fft": 4096,
        "hop_length": 256,
        "win_length": 4096,
        "n_iter": 512
    }

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=mel_params["sample_rate"],
        n_fft=mel_params["n_fft"],
        hop_length=mel_params["hop_length"],
        win_length=mel_params["win_length"],
        n_mels=mel_params["n_mels"],
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm='slaney',
        onesided=True,
        mel_scale="htk"
    )

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=spectro_params["n_fft"],
        hop_length=spectro_params["hop_length"],
        win_length=spectro_params["win_length"],
        center=True,
        pad_mode="reflect",
        power=2.0
    )

    isd = InstrumentSoundDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLE,
        device
    )

    spectrogram_to_signal = torchaudio.transforms.GriffinLim(
        n_fft=spectro_params["n_fft"],
        n_iter=spectro_params["n_iter"],
        hop_length=spectro_params["hop_length"],
        win_length=spectro_params["win_length"],
        power=2.0,
        # Using partial to modify the method default parameters so as to use the right device
        window_fn=partial(torch.hann_window, device=isd.device)
    )

    print(f"There are {len(isd)} samples in the dataset !")
    signal, label = isd[0]

    post = spectrogram_to_signal(signal)
    torchaudio.save("test_spectro_44100.wav", post.to("cpu"), SAMPLE_RATE)


