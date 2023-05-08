import torch
from typing import Tuple
from spotipy.client import Spotify
from pydub import AudioSegment
from io import BytesIO


def split_waveform(waveform: torch.Tensor, sr: int, secs_per_slice: int):

    slices = []

    for i in range(waveform.shape[-1]//sr//secs_per_slice):
        slices.append(waveform[i*sr*secs_per_slice:][:sr*secs_per_slice].unsqueeze(0))
        # let's make this a (1, N) Tensor... Will be useful when stacking for a single pass into the model

    return slices


def get_tracks_from_playlist(client: Spotify, playlist_id: str) -> Tuple[int, list]:
    """Gets the track ID's of tracks in a playlist

    Args:
        playlist_id (str): ID of playlist.

    Returns:
        Tuple[int, list]: (Number of tracks, List of track ID's)
    """

    playlist = client.playlist(playlist_id, market="US")
    tracks = [track["track"] for track in playlist["tracks"]["items"]]

    return len(tracks), tracks


def mp3_bytes_to_wav_bytes(mp3: BytesIO) -> BytesIO:
    """Convert mp3 bytes to wav bytes

    Args:
        mp3 (BytesIO): mp3 bytes

    Returns:
        BytesIO: wav bytes
    """
    s = AudioSegment.from_mp3(mp3).set_frame_rate(22050).set_channels(1)
    wav = BytesIO()
    s.export(wav, format="wav")
    return wav


def cosine_similarity(x, y):
    x = x.squeeze()
    y = y.squeeze()

    dot = x@y
    norms = torch.norm(x)*torch.norm(y)
    return (dot/norms).item()
