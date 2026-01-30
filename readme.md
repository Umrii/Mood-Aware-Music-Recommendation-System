# 🎧 Mood-Aware Music Recommendation System

A content-based music recommendation system that groups songs into interpretable mood clusters using Spotify audio features and serves mood-consistent song recommendations through a lightweight REST API.

This project is designed to mirror **real-world recommender system architecture**:  
model training and analysis are performed offline, while the online service only consumes precomputed results for fast and reproducible recommendations.

---

## 📌 Problem Statement

Music recommendation systems often need to operate under **cold-start conditions**, where little or no user history is available.  
In such cases, **audio features** provide a powerful signal to infer mood, energy, and listening context.

This project answers the question:

> *How can we recommend mood-consistent songs when user behavior data is limited or unavailable?*

---

## 🧠 Approach Overview

The system is built in two clearly separated stages:

### 1️⃣ Offline Mood Modeling (`moods.py`)
- Learn structure from Spotify audio features
- Cluster songs into mood groups using K-Means
- Assign human-interpretable mood labels post-hoc
- Validate clustering quality and compare against random baselines
- Persist all results to disk

### 2️⃣ Online Recommendation Service (`app.py`)
- Load precomputed song moods at startup
- Accept a song ID as input
- Recommend other songs from the same mood cluster
- Rank results by popularity for playlist-friendly output

No model training happens at request time.

---

## 📊 Dataset

The project uses a Spotify audio features dataset containing:

- Track metadata (artist, title, popularity)
- Audio features such as:
  - `valence`
  - `energy`
  - `danceability`
  - `tempo`
  - `acousticness`
  - `speechiness`
  - `instrumentalness`

These features are publicly documented by Spotify and are widely used in music recommendation research.

---

## 🎼 Mood Modeling Details

### Feature Set Used
```text
valence, energy, danceability, acousticness,
loudness, speechiness, tempo, instrumentalness
