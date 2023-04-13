import torch
from torch.utils.data import Dataset
from pipelines import AudioPipeline
import librosa
import pandas as pd
import os
from typing import Tuple
import yaml
from utils import split_waveform


class AudioData(Dataset):
    def __init__(self, train=True, data_dir="../data/raw", audio_dir="../data/audio", device="cuda"):
        self.data_dir = data_dir
        self.audio_dir = audio_dir
        self.device = device

        self.pipeline = AudioPipeline().to(device)

        self.df = pd.read_csv(os.path.join(data_dir, "train.csv" if train else "test.csv"))

        vals = yaml.safe_load(open("../config/data.yml", "r"))

        self.max = vals["SPECTROGRAM_MAX"]
        self.min = vals["SPECTROGRAM_MIN"]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        track_id = row["id"]

        waveform, sr = librosa.load(os.path.join(self.audio_dir, track_id) + ".mp3")

        waveform = torch.from_numpy(waveform).to(self.device)

        slices = split_waveform(waveform, sr=sr, secs_per_slice=3)

        y = []

        for slice in slices:
            x = self.pipeline(slice, sr).to(self.device)

            x = (x-self.min)/(self.max-self.min)

            y.append(x)

        if len(y) < 9:
            if len(y) == 0:
                y = [torch.zeros((128, 130)).to("cuda")]
            y += [y[-1]]*(9-len(y))

        return y


    def __len__(self):
        return len(self.df)