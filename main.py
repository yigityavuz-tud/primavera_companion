#!/usr/bin/env python3
import os
import argparse
import sys

# Add the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import load_and_explore_data, preprocess_playlist_data, analyze_genres
from src.modeling import train_and_evaluate_models, predict_and_rank_artists, analyze_artist_overlap
from src.visualization import plot_artist_distribution, plot_feature_importance
from src.utils import save_results, create_html_result

def main():
    parser = argparse.ArgumentParser(description='Primavera Sound Festival Artist Recommendation')
    parser.add_argument('--my-playlist', type=str, help='Path to your personal playlist CSV file')
    parser.add_argument('--primavera-playlist', type=str, help='Path to Primavera lineup playlist CSV file')
    parser.add_argument('--output-dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--min-artist-frequency', type=int, default=5, 
                        help='Minimum number of tracks an artist must have to be included (for Primavera data)')
    parser.add_argument('--top-n', type=int, default=30, help='Number of top artists to chart')
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

    # Step 7: Save results
    csv_path, json_path = save_results(ranked_artists, output_dir=args.output_dir)

    # Step 8: Create visualizations
    chart_path = plot_artist_distribution(ranked_artists, top_n=args.top_n, output_dir=args.output_dir)
    
    feature_names = train_data.drop(['Artist', 'Track_Count'], axis=1).columns
    feature_chart_path = plot_feature_importance(model_results, feature_names, output_dir=args.output_dir)

    # Step 9: Create HTML result
    html_path = create_html_result(ranked_artists, output_path=os.path.join(args.output_dir, "recommendations.html"))

    print("\n===== Recommendation Process Complete =====")
    print(f"Results saved to: {args.output_dir}")
    print(f"Full recommendation list: {csv_path}")
    print(f"Visualization: {chart_path}")
    print(f"HTML Report: {html_path}")
    print("\nEnjoy your personalized Primavera Sound schedule!")

if __name__ == "__main__":
    main()