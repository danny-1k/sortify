import yaml
import pandas as pd
import spotipy
import os
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
import requests

from concurrent.futures import ThreadPoolExecutor


class DataCollector:
    def __init__(self, metadata_save_dir="../data/raw/", audio_save_dir="../data/audio") -> None:
        self.metadata_save_dir = metadata_save_dir
        self.audio_save_dir = audio_save_dir

        if not os.path.exists(self.metadata_save_dir):
            os.makedirs(self.metadata_save_dir)

        if not os.path.exists(self.audio_save_dir):
            os.makedirs(self.audio_save_dir)


        auth = yaml.safe_load(open("../config/auth.yml", "r"))
        categories = yaml.safe_load(open("../config/categories.yml", "r"))
        config = yaml.safe_load(open("../config/collector.yml", "r"))
    
        os.environ["SPOTIPY_CLIENT_ID"] = auth["SPOTIFY_CLIENT_ID"]
        os.environ["SPOTIPY_CLIENT_SECRET"] = auth["SPOTIFY_CLIENT_SECRET"]

        self.client = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
        self.categories = [cat for cat in categories["categories"]]
        self.config = config

        self.data = {
            "id":[],
            "name":[],
            "preview_url": [],
            "category": [],
            "duration_ms": [],
        }

    def collect_data(self):
        limit = 50 # number of results per query

        for category in tqdm(self.categories):

            song_count = 0

            playlists = []


            playlist = self.client.category_playlists(category, limit=limit)["playlists"]
            total_number_of_playlists = playlist["total"]
            playlist_items = playlist["items"]

            playlists.extend(playlist_items)


            for round in range(limit, total_number_of_playlists, limit):
                if round > self.config["MAX_PLAYLISTS_SEARCH"]:
                    break

                playlist = self.client.category_playlists(category, limit=limit, offset=round//limit)["playlists"]
                playlist_items = playlist["items"]

                playlists.extend(playlist_items)


            for playlist in playlists:
                if song_count >= self.config["MAX_SONGS_PER_CATEGORY"]:
                    break

                if isinstance(playlist, dict):
                    tracks = playlist["tracks"]
                    tracks_id = tracks["href"].split("/")[-2]
                    tracks_playlist = self.client.playlist(tracks_id, market="US")["tracks"]["items"]

                    for track in tracks_playlist:
                        track = track["track"]

                        try:

                            if track["id"] not in self.data["id"]:
                                self.data["id"].append(track["id"])
                                self.data["name"].append(track["name"])
                                self.data["preview_url"].append(track["preview_url"])
                                self.data["category"].append(category)
                                self.data["duration_ms"].append(track["duration_ms"])

                                song_count += 1

                        except Exception as e:
                            print("could not get data out of this track... skipping")
                            continue

        # Save metadata as csv

        df = pd.DataFrame(self.data)

        df.to_csv(os.path.join(self.metadata_save_dir, "metadata.csv"))


    def download_audio_from_metadata(self):
        df = pd.read_csv(os.path.join(self.metadata_save_dir, "metadata.csv"))
        
        print(f"GOT {len(df)} data points")
        
        df = df.dropna(axis=0)

        print(f"DROPPED NULL ROWS NOW : {len(df)}")

        df.to_csv(os.path.join(self.metadata_save_dir, "metadata.csv"))

        print("Saved")

        pool = ThreadPoolExecutor(max_workers=16)


        def download_audio(data):
            [id, url] = data

            if os.path.exists(os.path.join(self.audio_save_dir, f"{id}.mp3")):
                print("Skipping... Already downloaded")

            else:
                r = requests.get(url)
                open(os.path.join(self.audio_save_dir, f"{id}.mp3"), "wb").write(r.content)


        pool.map(download_audio, df[["id", "preview_url"]].values.tolist())

        # for idx in tqdm(range(len(df))):
        #     item = df.iloc[idx]
        #     id = item["id"]
        #     preview_url = item["preview_url"]

        #     if str(preview_url) == "nan":
        #         continue

        #     if os.path.exists(os.path.join(self.audio_save_dir, f"{id}.mp3")):
        #         continue

        #     r = requests.get(preview_url)
            

        #     open(os.path.join(self.audio_save_dir, f"{id}.mp3"), "wb").write(r.content)

        pool.shutdown()


if __name__ == "__main__":
    collector = DataCollector()
    # collector.collect_data()

    print("started downloading audio")
    
    collector.download_audio_from_metadata()