#!/usr/bin/env python3
import os
import argparse
import sys

# Add the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import load_and_explore_data, preprocess_playlist_data, analyze_genres
from src.modeling import train_and_evaluate_models, predict_and_rank_artists, analyze_artist_overlap

def main():
    parser = argparse.ArgumentParser(description='Primavera Sound Festival Artist Recommendation')
    parser.add_argument('--my-playlist', type=str, required=True, help='Path to your personal playlist CSV file')
    parser.add_argument('--primavera-playlist', type=str, required=True, help='Path to Primavera lineup playlist CSV file')
    parser.add_argument('--output', type=str, default='primavera_recommendations.csv', help='Output CSV file path')
    parser.add_argument('--min-artist-frequency', type=int, default=5, 
                      help='Minimum number of tracks an artist must have to be included (for Primavera data)')
    args = parser.parse_args()

    print("===== Primavera Sound Artist Recommendation System =====")

    # Step 1: Load and explore data
    my_playlist, primavera_playlist = load_and_explore_data(
        my_playlist_path=args.my_playlist,
        primavera_playlist_path=args.primavera_playlist
    )

    # Step 2: Analyze genres and find shared ones
    top_shared_genres = analyze_genres(my_playlist, primavera_playlist)

    # Step 3: Process both datasets
    train_data = preprocess_playlist_data(
        my_playlist, 
        is_training=True, 
        shared_genres=top_shared_genres
    )

    test_data = preprocess_playlist_data(
        primavera_playlist, 
        is_training=False, 
        shared_genres=top_shared_genres,
        min_artist_frequency=args.min_artist_frequency
    )

    # Step 4: Train and evaluate models
    model_results = train_and_evaluate_models(train_data)

    # Step 5: Predict and rank artists
    ranked_artists, predicted_test_data = predict_and_rank_artists(test_data, model_results)

    # Step 6: Analyze artist overlap
    ranked_artists = analyze_artist_overlap(
        ranked_artists, my_playlist, primavera_playlist
    )

    # Step 7: Save results to CSV
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ranked_artists.to_csv(args.output, index=False)
    
    print(f"\nRecommendations saved to: {args.output}")
    print("\nEnjoy your personalized Primavera Sound schedule!")

if __name__ == "__main__":
    main()