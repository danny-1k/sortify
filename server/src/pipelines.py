import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram, Resample, AmplitudeToDB
import numpy as np
import requests
import io
from src.utils import mp3_bytes_to_wav_bytes, split_waveform
import torchaudio


class AudioPipeline(nn.Module):
    def __init__(self, sampling_freq: int = 22050) -> None:
        super().__init__()
        self.sampling_freq = sampling_freq
        self.spectrogram = MelSpectrogram(
            n_fft=1024,
            n_mels=128,
            sample_rate=self.sampling_freq,
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


class UrlToBytesToLatent:
    def __init__(self, net, pipeline, spec_min=-100, spec_max=49.23) -> None:
        self.net = net
        self.pipeline = pipeline
        self.spec_min = spec_min
        self.spec_max = spec_max

    def __call__(self, url: str) -> torch.Tensor:
        mp3_bytes = io.BytesIO(requests.get(url).content)
        wav_bytes = mp3_bytes_to_wav_bytes(mp3_bytes)
        waveform, sr = torchaudio.load(wav_bytes)
        waveform = waveform.squeeze()

        slices = split_waveform(waveform, sr=sr, secs_per_slice=3)

        slices = torch.cat(slices, dim=0)  # (N, n)

        spectrogram = self.pipeline(slices, sr).unsqueeze(1)  # (N,1,W,H)
        spectrogram = (spectrogram - self.spec_min) / \
            (self.spec_max - self.spec_min)  # standardize
        
        latents = self.net.encode(spectrogram) # (N, latent_dim)
        latent = latents.mean(axis=0, keepdim=True)

        return latent
