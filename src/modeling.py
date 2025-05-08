import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


def train_and_evaluate_models(train_data):
    """Train a gradient boosting regression model"""
    # Separate features and target
    X = train_data.drop(['Artist', 'Track_Count'], axis=1)
    y = train_data['Track_Count']
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Initialize Gradient Boosting model
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Train on full dataset for prediction
    gb_model.fit(X, y)
    
    # Store results
    results = {
        'Gradient Boosting': {
            'model': gb_model
        }
    }
    
    return results


def predict_and_rank_artists(test_data, model_results):
    """Predict scores for test artists and rank them"""
    # Prepare test features
    X_test = test_data.drop(['Artist'], axis=1)
    
    # Fill any remaining NaN values
    X_test = X_test.fillna(0)
    
    # Use the Gradient Boosting model for prediction
    gb_model = model_results['Gradient Boosting']['model']
    
    # Predict scores
    test_data_copy = test_data.copy()
    test_data_copy['Predicted_Score'] = gb_model.predict(X_test)
    
    # Ensure no negative scores
    test_data_copy['Predicted_Score'] = test_data_copy['Predicted_Score'].clip(lower=0)
    
    # Rank artists by predicted score
    ranked_artists = test_data_copy[['Artist', 'Predicted_Score']].sort_values(
        'Predicted_Score', ascending=False
    ).reset_index(drop=True)
    
    # Add rank column
    ranked_artists['Rank'] = ranked_artists.index + 1
    
    # Reorder columns
    ranked_artists = ranked_artists[['Rank', 'Artist', 'Predicted_Score']]
    
    return ranked_artists, test_data_copy


def analyze_artist_overlap(ranked_artists, my_playlist_df, primavera_playlist_df):
    """Analyze overlap between personal playlist and Primavera artists"""
    # Extract all artists from personal playlist
    my_artists = set(my_playlist_df['Artist Name(s)'].str.split(', ').explode().unique())
    
    # Check which Primavera artists are in my playlist
    ranked_artists['In_My_Playlist'] = ranked_artists['Artist'].apply(
        lambda x: 1 if x in my_artists else 0
    )
    
    # Count overlap
    overlap_count = ranked_artists['In_My_Playlist'].sum()
    
    # Adjust scores based on overlap
    if overlap_count > 0:
        # Find maximum predicted score for normalization
        max_score = ranked_artists['Predicted_Score'].max()
        
        # Apply weight to overlap (30% of the score comes from playlist overlap)
        overlap_weight = 0.3
        
        # Create adjusted score
        ranked_artists['Adjusted_Score'] = (
            (1 - overlap_weight) * ranked_artists['Predicted_Score'] + 
            overlap_weight * max_score * ranked_artists['In_My_Playlist']
        )
        
        # Re-rank based on adjusted score
        ranked_artists = ranked_artists.sort_values(
            'Adjusted_Score', ascending=False
        ).reset_index(drop=True)
        
        # Update rank
        ranked_artists['Adjusted_Rank'] = ranked_artists.index + 1
    
    return ranked_artists