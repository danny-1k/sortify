import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB
import numpy as np


class AudioPipeline(nn.Module):
    def __init__(self, sampling_freq: int=22050) -> None:
        super().__init__()
        self.sampling_freq = sampling_freq
        self.spectrogram = MelSpectrogram(
            n_fft=1024,
            n_mels=128,
            sample_rate = self.sampling_freq,
        )
        self.toDB = AmplitudeToDB()

    def forward(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        resampler = self.create_resample(sr)

        waveform = resampler(waveform)

        spec = self.spectrogram(waveform)
        spec = self.toDB(spec)

        return spec

    def create_resample(self, in_sr: int) -> Resample:
        r = Resample(in_sr, self.sampling_freq)
        return r

if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt
    import torch

    waveform, sr = librosa.load("../data/audio/57225qPFGFLt8MTI10FohE.mp3")
    pipeline = AudioPipeline().cuda()

    spec = pipeline(waveform, sr,).numpy()

    plt.imshow(spec)

    plt.savefig("spec.png")

    # print(spec.max())