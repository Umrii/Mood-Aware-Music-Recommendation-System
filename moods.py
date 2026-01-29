import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Loading Spotify dataset...")
df = pd.read_csv('SpotifyAudioFeaturesApril2019.csv')


# Remove duplicates and handle missing values
df = df.drop_duplicates(subset=['track_id'])
df = df.dropna()


print(f"Dataset shape: {df.shape}")
print(f"\nDataset columns: {df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")


mood_features = ['valence', 'energy', 'danceability', 'acousticness', 
                 'loudness', 'speechiness', 'tempo', 'instrumentalness']


# Normalize features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[mood_features])