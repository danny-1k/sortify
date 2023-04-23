# Sortify Backend

The Flask server and the sorting algorithm is implemented here.

## Algorithm

In the playlist (collection of `tracks`), we choose one track to be an `anchor`.

The anchor is going to be the first track in the sorted playlist and the track that will be used to compare other tracks for similarity.

The chosen `anchor` track is not constant.

The idea is as the tracks in the playlist evolve, losing similarity with the main anchor, a new anchor is chosen to base of the similarity comparisons off.

 **cosine similarity** is used to determine the similarity of the latent features of the track preview sample. The trained autoencoder is used for compressing/obtaining these latent features

To determine the different groups of similar tracks, we take the mean and standard deviation of all the cosine similarities of the tracks being compared with the current anchor track, then, only consider tracks that have the similarity score > `mu - sigma`.

The grouped tracks are sorted in decreasing cosine similarity.

The "next anchor" is chosen from the least dissimilar track with the current anchor track
