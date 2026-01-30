from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI()

# Load dataset once at startup
df = pd.read_csv("spotify_with_moods.csv")

@app.get("/recommend/song/{track_id}")
def recommend_similar_songs(track_id: str, limit: int = 10):
    # 1. Find the song
    song = df[df["track_id"] == track_id]

    if song.empty:
        raise HTTPException(status_code=404, detail="Song not found")

    # 2. Get its cluster
    cluster = song.iloc[0]["mood_cluster"]

    # 3. Filter songs in same cluster
    similar_songs = df[
        (df["mood_cluster"] == cluster) &
        (df["track_id"] != track_id)
    ]

    # 4. Rank by popularity (Spotify-like default)
    similar_songs = similar_songs.sort_values(
        by="popularity", ascending=False
    )

    # 5. Return top N
    return similar_songs.head(limit).to_dict(orient="records")
