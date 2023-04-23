import torch
from src.inference import cosine_similarity
from src.pipelines import UrlToBytesToLatent


class Track:
    def __init__(self, track_id, track_url: str, is_anchor=False):
        self.track_id = track_id
        self.is_anchor = is_anchor
        self.track_url = track_url
        self.latent = None
        self.anchor_similarity = None

    def get_latent(self, pipeline: UrlToBytesToLatent) -> torch.Tensor:
        self.latent = pipeline(self.track_url)

    def compare(self, track):
        similarity = cosine_similarity(self.latent, track.latent)

        return similarity

    def __eq__(self, track) -> bool:
        return self.track_url == track.track_url

    def __lt__(self, track) -> bool:
        return self.anchor_similarity <= track.anchor_similarity

    def __gt__(self, track) -> bool:
        return self.anchor_similarity >= track.anchor_similarity


class Playlist:
    def __init__(self, pipeline: UrlToBytesToLatent, tracks) -> None:
        self.pipeline = pipeline

        if tracks == []:
            self.tracks = []

        elif isinstance(tracks[0], str):  # assuming a list of urls
            self.tracks = [Track(url) for url in tracks]

        elif isinstance(tracks[0], Track):
            self.tracks = tracks

        else:
            raise ValueError("Invalid value for tracks")

        [track.get_latent(pipeline)
         for track in self.tracks if isinstance(track.latent, type(None))]

    def get_anchor(self) -> Track:
        for track in self.tracks:
            if track.is_anchor:
                return track

    def set_anchor(self, to_set_track: Track) -> None:
        for track in self.tracks:
            if track.track_url == to_set_track.track_url if isinstance(to_set_track, Track) else to_set_track:
                track.is_anchor = True

    def add_track(self, track: Track) -> None:
        if track not in self.tracks:
            track.get_latent(self.pipeline)
            self.tracks.append(track)

    def remove_track(self, to_remove_track: Track) -> None:
        for idx, track in enumerate(self.tracks):
            # `to_remove_track` could be a `Track` object or a `track_url` string
            # check if the tracks have the same track_url
            if track.track_url == to_remove_track.track_url if isinstance(to_remove_track, Track) else to_remove_track:
                del self.tracks[idx]
                break

    def __add__(self, track) -> float:
        if hasattr(track, "anchor_similarity") and hasattr(self, "anchor_similarity"):
            return self.anchor_similarity + track.anchor_similarity

    def __len__(self) -> int:
        return len(self.tracks)


class Algorithm:
    def __init__(self, playlist: Playlist) -> None:
        self.playlist = playlist

    def sort(self):

        # New playlist with sorted tracks
        sorted_playlist = Playlist(pipeline=self.playlist.pipeline, tracks=[])

        while len(self.playlist) != 0:

            anchor = self.playlist.get_anchor()
            tracks_similarity = []

            for track in self.playlist.tracks:
                similarity = cosine_similarity(track.latent, anchor.latent)
                track.anchor_similarity = similarity
                tracks_similarity.append(similarity)

            # Find the mean similarity and standard deviation with the chosen anchor
            # we can use this to determine "outliers" so we can choose new anchors and sort better

            tracks_similarity = torch.Tensor(tracks_similarity)
            tracks_similarity_mean = tracks_similarity.mean().item()
            tracks_similarity_std = tracks_similarity.std().item()

            similarity_cuttoff = tracks_similarity_mean - tracks_similarity_std
            # tracks that have a similarity greater than mu - sigma will chosen

            tracks_to_sort = []
            dissimilar_tracks = []

            for track in self.playlist.tracks:
                if track.anchor_similarity > similarity_cuttoff:
                    tracks_to_sort.append(track)
                    # remove track from the playlist
                    self.playlist.remove_track(track)
                else:
                    dissimilar_tracks.append(track)

            # sort from most similar to least similar
            tracks_to_sort.sort(reverse=True)
            # sort from least dissimilar to most dissimilar
            dissimilar_tracks.sort(reverse=True)

            [sorted_playlist.add_track(track) for track in tracks_to_sort]

            if len(dissimilar_tracks) == 0:  # end
                if len(self.playlist) == 0:
                    break

                elif len(self.playlist) == 1:
                    last_track = self.playlist.tracks[-1]

                    sorted_playlist.add_track(last_track)
                    self.playlist.remove_track(last_track)

                    break
                else:
                    anchor = self.playlist.tracks[0]
                    self.playlist.set_anchor(anchor)

            else:

                # The track least dissimilar with the previous anchor
                anchor = dissimilar_tracks[0]
                # we're doing this because we want the tracks to be sorted with similarity gradually decreasing

                self.playlist.set_anchor(anchor)  # set the new anchor

        return sorted_playlist
