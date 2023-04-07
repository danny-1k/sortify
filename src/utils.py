import torch

def split_waveform(waveform:torch.Tensor, sr:int, secs_per_slice:int):

    slices = []

    for i in range(waveform.shape[-1]//sr//secs_per_slice):
        slices.append(waveform[i*sr*secs_per_slice:][:sr*secs_per_slice])

    return slices