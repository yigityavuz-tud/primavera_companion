import os
import pandas as pd
import numpy as np


def load_and_explore_data(my_playlist_path=None, primavera_playlist_path=None):
    """Load the data files and perform initial exploration"""
    # Check if data exists in expected location
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} directory not found. Using current directory.")
        data_dir = "."
    
    # Find CSV files if paths not provided
    if not my_playlist_path:
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                if 'my' in file.lower() or 'personal' in file.lower() or 'taste' in file.lower() or 'matrix' in file.lower():
                    my_playlist_path = os.path.join(data_dir, file)
                    break
    
    if not primavera_playlist_path:
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                if 'primavera' in file.lower() or 'festival' in file.lower():
                    primavera_playlist_path = os.path.join(data_dir, file)
                    break
    
    if not my_playlist_path:
        raise ValueError("Personal playlist CSV not found. Please upload your playlist CSV file.")
    
    if not primavera_playlist_path:
        raise ValueError("Primavera playlist CSV not found. Please check the data directory.")
    
    # Load datasets
    print(f"Loading personal playlist from: {my_playlist_path}")
    my_playlist = pd.read_csv(my_playlist_path)
    
    print(f"Loading Primavera playlist from: {primavera_playlist_path}")
    primavera_playlist = pd.read_csv(primavera_playlist_path)
    
    # Basic exploration
    print("\n----- Personal Playlist Summary -----")
    print(f"Number of tracks: {len(my_playlist)}")
    print(f"Number of unique artists: {my_playlist['Artist Name(s)'].str.split(', ').explode().nunique()}")
    print(f"Columns available: {my_playlist.columns.tolist()}")
    
    print("\n----- Primavera Playlist Summary -----")
    print(f"Number of tracks: {len(primavera_playlist)}")
    print(f"Number of unique artists: {primavera_playlist['Artist Name(s)'].str.split(', ').explode().nunique()}")
    print(f"Columns available: {primavera_playlist.columns.tolist()}")
    
    return my_playlist, primavera_playlist


def preprocess_playlist_data(playlist_df, is_training=True, shared_genres=None, return_genres=False, min_artist_frequency=None):
    """
    Preprocess playlist data for artist-level aggregation.
    - Explode artist collaborations into separate rows
    - Extract and encode genres
    - Aggregate numerical features by artist
    - Filter out infrequent artists (for Primavera data)
    
    Args:
        playlist_df: DataFrame containing playlist data
        is_training: Whether this is training data (True) or test data (False)
        shared_genres: List of shared genres to use (for test data)
        return_genres: Whether to return genre frequency info (True for initial training)
        min_artist_frequency: Minimum frequency for artists to be included (None = no filtering)
    """
    print(f"\n----- Preprocessing {'training' if is_training else 'test'} data -----")
    
    # Make a copy to avoid modifying the original DataFrame
    df = playlist_df.copy()
    
    # Check for needed columns and handle missing ones
    required_columns = [
        'Artist Name(s)', 'Genres', 'Popularity', 'Danceability', 'Energy', 
        'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 
        'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Time Signature'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Adding dummy column.")
            if col in ['Key', 'Mode', 'Time Signature']:
                df[col] = 0  # Default for categorical
            else:
                df[col] = np.nan  # Default for numerical
    
    # Handle missing values in numeric columns
    numeric_cols = [
        'Popularity', 'Danceability', 'Energy', 'Loudness', 
        'Speechiness', 'Acousticness', 'Instrumentalness', 
        'Liveness', 'Valence', 'Tempo'
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            median_val = df[col].median()
            print(f"Filling {df[col].isna().sum()} missing values in '{col}' with median: {median_val:.2f}")
            df[col].fillna(median_val, inplace=True)
    
    # Convert categorical columns to integers to avoid type issues
    for col in ['Key', 'Mode', 'Time Signature']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Explode artists to create one row per artist
    print("Exploding multiple artists in collaborations...")
    df['Artist Name(s)'] = df['Artist Name(s)'].astype(str)
    
    # Handle different separators for artist names (comma or comma+space)
    artist_lists = []
    for artists_str in df['Artist Name(s)']:
        if pd.isna(artists_str) or artists_str == '' or artists_str.lower() == 'nan':
            artist_lists.append([])
        else:
            if ', ' in artists_str:
                artist_lists.append([a.strip() for a in artists_str.split(', ') if a.strip()])
            else:
                artist_lists.append([a.strip() for a in artists_str.split(',') if a.strip()])
    
    df['Artist_List'] = artist_lists
    
    # Explode the dataframe on the artist list
    exploded_artists = df.explode('Artist_List')
    
    # Rename and clean up
    exploded_artists.rename(columns={'Artist_List': 'Artist'}, inplace=True)
    
    # Handle empty or NaN artist names
    exploded_artists = exploded_artists[exploded_artists['Artist'].notna() & (exploded_artists['Artist'] != '')]
    
    # Filter out infrequent artists if min_artist_frequency is specified (for Primavera data)
    if min_artist_frequency is not None and min_artist_frequency > 0:
        artist_counts = exploded_artists['Artist'].value_counts()
        frequent_artists = artist_counts[artist_counts >= min_artist_frequency].index
        
        original_count = exploded_artists['Artist'].nunique()
        exploded_artists = exploded_artists[exploded_artists['Artist'].isin(frequent_artists)]
        
        new_count = exploded_artists['Artist'].nunique()
        print(f"Filtered out {original_count - new_count} infrequent artists")
        print(f"Keeping {new_count} main artists with at least {min_artist_frequency} tracks")
        
        # If we filtered out all artists, that's a problem
        if len(exploded_artists) == 0:
            print("WARNING: All artists were filtered out. Reducing min_artist_frequency.")
            # Try with a lower threshold
            min_artist_frequency = 1
            exploded_artists = df.explode('Artist_List')
            exploded_artists.rename(columns={'Artist_List': 'Artist'}, inplace=True)
            exploded_artists = exploded_artists[exploded_artists['Artist'].notna() & (exploded_artists['Artist'] != '')]
    
    # Process genres - improved handling for different formats
    print("Processing genres...")
    
    # Check if Genres column exists, create it if it doesn't
    if 'Genres' not in exploded_artists.columns:
        print("Warning: 'Genres' column not found. Creating empty column.")
        exploded_artists['Genres'] = ''
    
    # Ensure Genres column is string type
    exploded_artists['Genres'] = exploded_artists['Genres'].astype(str)
    
    # Handle different genre separators (comma or comma+space)
    # Some may use ',', others may use ', '
    genre_lists = []
    for genres_str in exploded_artists['Genres']:
        # Handle NaN, empty strings, and 'nan' string
        if pd.isna(genres_str) or genres_str == '' or genres_str.lower() == 'nan':
            genre_lists.append([])
        else:
            # Try splitting by comma and space, if that doesn't work, try just comma
            if ', ' in genres_str:
                genre_lists.append([g.strip() for g in genres_str.split(', ') if g.strip()])
            else:
                genre_lists.append([g.strip() for g in genres_str.split(',') if g.strip()])
    
    exploded_artists['Genre_List'] = genre_lists
    
    # Calculate genre frequencies for training data
    if is_training and return_genres:
        # Flatten all genre lists
        all_genres_flat = [genre for sublist in genre_lists for genre in sublist if genre]
        
        # Count genre frequencies
        genre_counts = pd.Series(all_genres_flat).value_counts()
        
        print(f"Found {len(genre_counts)} unique genres in training data")
        print(f"Top 5 genres: {genre_counts.head(5).to_dict()}")
    
    # If shared_genres is provided, use only those genres
    # Otherwise, use all genres if in training mode
    if shared_genres is not None:
        genres_to_use = shared_genres
        print(f"Using {len(genres_to_use)} shared genres from both datasets")
    elif is_training:
        # Extract all unique genres
        all_genres = set()
        for genre_list in genre_lists:
            all_genres.update(genre_list)
        
        # Remove any empty strings
        if '' in all_genres:
            all_genres.remove('')
        
        genres_to_use = all_genres
        print(f"Using all {len(genres_to_use)} unique genres in training data")
    else:
        # This should not happen if code is used correctly
        print("WARNING: No shared_genres provided for test data. Using empty set.")
        genres_to_use = set()
    
    # PERFORMANCE IMPROVEMENT: Create all genre columns at once using list comprehension and concat
    print("Creating genre feature columns efficiently...")
    
    # Create a dictionary for genre encoding
    genre_dfs = []
    
    # Convert list of genres to use to a set for faster lookups
    genres_to_use_set = set(genres_to_use)
    
    # For each genre in the set, create a Series that will be a column
    for genre in genres_to_use_set:
        genre_col = exploded_artists['Genre_List'].apply(lambda x: 1 if genre in x else 0)
        genre_col.name = f'Genre_{genre}'
        genre_dfs.append(genre_col)
    
    # If we have genre columns, concat them all at once to the main DataFrame
    if genre_dfs:
        # Make a copy to defragment the DataFrame
        # This is faster than repeatedly adding columns to the original DataFrame
        exploded_artists = exploded_artists.copy()
        
        # Concatenate all genre columns at once
        genre_df = pd.concat(genre_dfs, axis=1)
        exploded_artists = pd.concat([exploded_artists, genre_df], axis=1)
    
    # Drop the temporary Genre_List column
    exploded_artists.drop('Genre_List', axis=1, inplace=True)
    
    # Count tracks per artist (for training data)
    if is_training:
        print("Calculating artist frequency in personal playlist...")
        artist_frequency = exploded_artists['Artist'].value_counts().reset_index()
        artist_frequency.columns = ['Artist', 'Track_Count']
    
    # Aggregate numerical features by artist
    print("Aggregating features by artist...")
    
    # List of numerical features to aggregate
    num_features = [
        'Popularity', 'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo'
    ]
    
    # Create aggregation dictionary
    agg_dict = {}
    for feature in num_features:
        agg_dict[feature] = ['min', 'max', 'mean', 'var']
    
    # Add genre columns to aggregation (using max because they're 0/1)
    genre_cols = [col for col in exploded_artists.columns if col.startswith('Genre_')]
    for col in genre_cols:
        agg_dict[col] = 'max'  # Just use 'max' without renaming
        
    # Add categorical columns with mode aggregation
    for col in ['Key', 'Mode', 'Time Signature']:
        agg_dict[col] = lambda x: x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else 0
    
    # Perform aggregation
    artist_features = exploded_artists.groupby('Artist').agg(agg_dict)
    
    # Flatten multi-level column names, but preserve genre column names WITHOUT _max suffix
    new_columns = []
    for col in artist_features.columns:
        if col[0].startswith('Genre_') and col[1] == 'max':
            # For genre columns, just use the original name without the _max suffix
            new_columns.append(col[0])
        else:
            # For other columns, join the names
            new_columns.append('_'.join(col).strip())
    
    artist_features.columns = new_columns
    
    # Reset index to make 'Artist' a column
    artist_features.reset_index(inplace=True)
    
    # For training data, merge with track count
    if is_training:
        artist_features = artist_features.merge(artist_frequency, on='Artist', how='left')
        artist_features['Track_Count'].fillna(0, inplace=True)
        print(f"Created features for {len(artist_features)} artists with track count as target variable")
    else:
        print(f"Created features for {len(artist_features)} artists")
    
    # Return the processed data, and genre counts if requested
    if is_training and return_genres:
        return artist_features, genre_counts
    return artist_features


def analyze_genres(my_playlist, primavera_playlist):
    """Analyze and find shared genres between datasets"""
    print("\n----- Analyzing genres in both datasets -----")
    
    # First pass to get genres from personal playlist
    _, my_genres = preprocess_playlist_data(my_playlist, is_training=True, return_genres=True)
    
    # Do a light processing of Primavera data to get genres
    primavera_exploded = primavera_playlist.copy()
    primavera_exploded['Genres'] = primavera_exploded['Genres'].astype(str)
    
    # Extract genres from Primavera data
    primavera_genre_lists = []
    for genres_str in primavera_exploded['Genres']:
        if pd.isna(genres_str) or genres_str == '' or genres_str.lower() == 'nan':
            primavera_genre_lists.append([])
        else:
            if ', ' in genres_str:
                primavera_genre_lists.append([g.strip() for g in genres_str.split(', ') if g.strip()])
            else:
                primavera_genre_lists.append([g.strip() for g in genres_str.split(',') if g.strip()])
    
    # Flatten all genre lists from Primavera
    primavera_genres_flat = [genre for sublist in primavera_genre_lists for genre in sublist if genre]
    
    # Count genre frequencies in Primavera data
    primavera_genre_counts = pd.Series(primavera_genres_flat).value_counts()
    
    # Find genres that appear in both datasets
    my_genre_set = set(my_genres.index)
    primavera_genre_set = set(primavera_genre_counts.index)
    shared_genres = my_genre_set.intersection(primavera_genre_set)
    
    print(f"Found {len(shared_genres)} genres shared between datasets")
    
    # Get the top 20 most frequent genres from your playlist that also appear in Primavera data
    top_shared_genres = my_genres[my_genres.index.isin(shared_genres)].nlargest(20).index.tolist()
    
    print(f"Selected top 20 shared genres: {top_shared_genres}")
    
    return top_shared_genres