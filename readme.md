# Mood-Aware Playlist Recommendation System

## Overview
This system analyzes Spotify audio features to classify songs into 8 distinct mood categories and provides an interactive web interface for generating mood-based playlists.

## System Architecture

### 1. Data Processing & Mood Classification
The system uses **K-Means clustering** on the following audio features:
- **Valence**: Musical positiveness (0.0 = negative, 1.0 = positive)
- **Energy**: Intensity and activity (0.0 = calm, 1.0 = energetic)
- **Danceability**: Suitability for dancing (0.0-1.0)
- **Acousticness**: Confidence the track is acoustic (0.0-1.0)
- **Tempo**: BPM (beats per minute)
- **Loudness**: Overall loudness in decibels
- **Speechiness**: Presence of spoken words (0.0-1.0)
- **Instrumentalness**: Predicts if track has no vocals (0.0-1.0)
