import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

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



print("\nCreating mood clusters...")
n_clusters = 8  # Based on Human Moods, I have used 8 clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['mood_cluster'] = kmeans.fit_predict(scaled_features)


# Analyze each cluster to understand mood characteristics
mood_profiles = []
for i in range(n_clusters):
    cluster_data = df[df['mood_cluster'] == i][mood_features]
    profile = {
        'cluster': i,
        'size': len(cluster_data),
        'avg_valence': float(cluster_data['valence'].mean()),
        'avg_energy': float(cluster_data['energy'].mean()),
        'avg_danceability': float(cluster_data['danceability'].mean()),
        'avg_acousticness': float(cluster_data['acousticness'].mean()),
        'avg_tempo': float(cluster_data['tempo'].mean())
    }
    mood_profiles.append(profile)
    
# Sort clusters by valence and energy to assign mood labels
mood_profiles_sorted = sorted(mood_profiles, key=lambda x: (x['avg_valence'], x['avg_energy']))

mood_labels = [
    'Melancholic',      # Low valence, low energy
    'Sad',              # Low valence, medium-low energy
    'Anxious',          # Low valence, medium energy
    'Angry',            # Low valence, high energy
    'Calm',             # Medium valence, low energy
    'Peaceful',         # Medium-high valence, low energy
    'Happy',            # High valence, medium energy
    'Euphoric'          # High valence, high energy
]

# Map clusters to mood labels
cluster_to_mood = {}
for idx, profile in enumerate(mood_profiles_sorted):
    cluster_num = profile['cluster']
    cluster_to_mood[cluster_num] = {
        'mood': mood_labels[idx],
        'profile': profile
    }

# Add mood labels to dataframe
df['mood_label'] = df['mood_cluster'].map(lambda x: cluster_to_mood[x]['mood'])

# Save processed data
print("\nSaving processed data...")
df.to_csv('spotify_with_moods.csv', index=False)

# Create a sample dataset for the web app (to reduce file size)
print("\nCreating sample dataset for web app...")
sample_df = df.groupby('mood_label').apply(lambda x: x.nlargest(500, 'popularity')).reset_index(drop=True)
sample_df.to_csv('spotify_sample.csv', index=False)

# Save mood mapping and statistics
mood_stats = {}
for mood_label in mood_labels:
    mood_data = df[df['mood_label'] == mood_label]
    mood_stats[mood_label] = {
        'count': int(len(mood_data)),
        'avg_valence': float(mood_data['valence'].mean()),
        'avg_energy': float(mood_data['energy'].mean()),
        'avg_danceability': float(mood_data['danceability'].mean()),
        'avg_acousticness': float(mood_data['acousticness'].mean()),
        'avg_tempo': float(mood_data['tempo'].mean()),
        'popular_artists': mood_data['artist_name'].value_counts().head(5).to_dict()
    }

with open('mood_stats.json', 'w') as f:
    json.dump(mood_stats, f, indent=2)

print("\n=== Mood Distribution ===")
print(df['mood_label'].value_counts().sort_index())

print("\n=== Mood Characteristics ===")
for mood, stats in mood_stats.items():
    print(f"\n{mood}:")
    print(f"  Count: {stats['count']}")
    print(f"  Valence: {stats['avg_valence']:.3f}")
    print(f"  Energy: {stats['avg_energy']:.3f}")
    print(f"  Danceability: {stats['avg_danceability']:.3f}")

print("\nAnalysis complete!")