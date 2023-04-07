import torch
import librosa
import pandas as pd
from utils import split_waveform
from pipelines import AudioPipeline
from tqdm import tqdm


def calculate_stats():
    max_val = 0
    min_val = 0
    
    pipeline = AudioPipeline().to("cuda")

    ids = pd.read_csv("../data/raw/train.csv")["id"].values.tolist()

    for id in tqdm(ids):
        try:
            waveform, sr = librosa.load(f"../data/audio/{id}.mp3")
            waveform = torch.from_numpy(waveform).to("cuda")
            slices = split_waveform(waveform=waveform, sr=sr, secs_per_slice=3)

            for slice in slices:
                spec = pipeline(slice, sr)

                max_val = max(spec.max().item(), max_val)
                min_val = min(spec.min().item(), min_val)

        except EOFError:
            print(f"Could not get -> {id}")

    print(f"MAX -> {max_val} \n MIN -> {min_val}")

if __name__ == "__main__":
    calculate_stats()