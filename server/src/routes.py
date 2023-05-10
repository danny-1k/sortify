import os
import torch

from src import app
from src.utils import get_tracks_from_playlist
from src.models import AutoEncoder
from src.pipelines import AudioPipeline, UrlToBytesToLatent
from src.algorithm import Algorithm, Playlist, Track

from flask import abort, request, Response, jsonify, redirect

import requests

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials




spotify_client = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

net = AutoEncoder()
net.load_state_dict(torch.load("checkpoint/checkpoint.pt",
                    map_location=DEVICE)["state_dict"])
net.requires_grad_(False)
net.eval()
net.to(DEVICE)

audiopipeline = AudioPipeline()
audiopipeline.to(DEVICE)

tolatentpipeline = UrlToBytesToLatent(net=net, pipeline=audiopipeline)


@app.route("/")
def index():
    return "v1"


@app.route("/isalive")
def isalive():
    return Response(200)


@app.route("/", methods=["POST"])
def predict():
    json = request.get_json()

    anchor_id = json["anchor"]
    playlist_id = json["playlist"]

    with torch.no_grad():
        number_of_tracks, tracks = get_tracks_from_playlist(
            spotify_client, playlist_id)

        if number_of_tracks >= (int(os.environ.get("MAX_NUMBER_OF_TRACKS")) or 100):
            abort(400, "Number of tracks in the playlist exceeds the limit")

        track_ids = [track["id"] for track in tracks]
        track_preview_urls = [track["preview_url"] for track in tracks]

        tracks = zip(track_ids, track_preview_urls)

        playlist = Playlist(pipeline=tolatentpipeline, tracks=[])

        for id, preview_url in tracks:

            track = Track(track_id=id,
                          track_url=preview_url,
                          is_anchor=id == anchor_id)

            playlist.add_track(track)

        if isinstance(playlist.get_anchor(), type(None)):
            return abort(Response(400, "Provided Anchor Track does not exist in playlist..."))

        algorthm = Algorithm(playlist)

        sorted_playlist = algorthm.sort()

        return jsonify({
            "predictions": [track.track_id for track in sorted_playlist.tracks]
        })

@app.route("/inference")
def devilsh_ting():
    return redirect("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


@app.errorhandler(400)
def bad_request(error):
    return f"Could not complete Request. {error}"


@app.errorhandler(404)
def notfound(error):
    return f"Not found."
