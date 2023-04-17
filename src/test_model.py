import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from pipelines import AudioPipeline
import librosa
from utils import split_waveform
from models import AutoEncoder
import matplotlib.pyplot as plt

import yaml
vals = yaml.safe_load(open("../config/data.yml", "r"))

SPEC_MAX = vals["SPECTROGRAM_MAX"]
SPEC_MIN = vals["SPECTROGRAM_MIN"]


net = AutoEncoder()
net.to("cuda")
net.load_state_dict(torch.load("../checkpoints/20230413-221351/checkpoint.pt")["state_dict"])

waveform, sr = librosa.load("../data/audio/44SSviC4R1TkAdsyptjDpE.mp3")
waveform = torch.from_numpy(waveform).to("cuda")

slices = split_waveform(waveform, sr=sr, secs_per_slice=3)

pipeline = AudioPipeline().to("cuda")

with torch.no_grad():

    for idx, slice in enumerate(slices):
        x = pipeline(slice, sr).to("cuda")
        x_inp = (x-SPEC_MIN)/(SPEC_MAX-SPEC_MIN)

        reconstructed = net(x_inp.unsqueeze(0).unsqueeze(0)).squeeze()

        plt.imshow(x.cpu())
        plt.savefig(f"slice_{idx}.png")
        plt.close("all")

        reconstructed = (reconstructed*(SPEC_MAX-SPEC_MIN)) + SPEC_MIN

        plt.imshow(reconstructed.cpu())
        plt.savefig(f"slice_reconstructed_{idx}.png")