import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR


def train_and_evaluate_models(train_data):
    """Train multiple regression models and evaluate their performance"""
    print("\n----- Training and Evaluating Models -----")
    
    # Separate features and target
    X = train_data.drop(['Artist', 'Track_Count'], axis=1)
    y = train_data['Track_Count']
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Initialize models to try
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf')
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train with cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-cv_scores)
        
        # Train on full dataset for later prediction
        model.fit(X, y)
        
        # Store results
        results[name] = {
            'model': model,
            'cv_rmse_mean': rmse_scores.mean(),
            'cv_rmse_std': rmse_scores.std()
        }
        
        print(f"  {name} - RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
    
    # Find best model
    best_model_name = min(results, key=lambda k: results[k]['cv_rmse_mean'])
    print(f"\nBest performing model: {best_model_name} with RMSE: {results[best_model_name]['cv_rmse_mean']:.4f}")
    
    # Feature importance for tree-based models
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features (Random Forest):")
        print(feature_importance.head(10))
    
    return results


def predict_and_rank_artists(test_data, model_results):
    """Predict scores for test artists and rank them"""
    print("\n----- Predicting and Ranking Artists -----")
    
    # Prepare test features
    X_test = test_data.drop(['Artist'], axis=1)
    
    # Fill any remaining NaN values
    X_test = X_test.fillna(0)
    
    # Use the best model for prediction
    best_model_name = min(model_results, key=lambda k: model_results[k]['cv_rmse_mean'])
    best_model = model_results[best_model_name]['model']
    
    print(f"Using {best_model_name} for prediction...")
    
    # Predict scores
    test_data_copy = test_data.copy()
    test_data_copy['Predicted_Score'] = best_model.predict(X_test)
    
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
    
    print("\nTop 20 recommended artists from Primavera:")
    print(ranked_artists.head(20))
    
    return ranked_artists, test_data_copy


def analyze_artist_overlap(ranked_artists, my_playlist_df, primavera_playlist_df):
    """Analyze overlap between personal playlist and Primavera artists"""
    print("\n----- Analyzing Artist Overlap -----")
    
    # Extract all artists from personal playlist
    my_artists = set(my_playlist_df['Artist Name(s)'].str.split(', ').explode().unique())
    
    # Check which Primavera artists are in my playlist
    ranked_artists['In_My_Playlist'] = ranked_artists['Artist'].apply(
        lambda x: 1 if x in my_artists else 0
    )
    
    # Count overlap
    overlap_count = ranked_artists['In_My_Playlist'].sum()
    print(f"Found {overlap_count} Primavera artists in your personal playlist")
    
    # Highlight artists in common
    if overlap_count > 0:
        print("\nPrimavera artists in your playlist:")
        overlap_artists = ranked_artists[ranked_artists['In_My_Playlist'] == 1]
        print(overlap_artists[['Rank', 'Artist', 'Predicted_Score']])
    
    # Adjust scores based on overlap
    if overlap_count > 0:
        # Find maximum predicted score for normalization
        max_score = ranked_artists['Predicted_Score'].max()
        
        # Apply weight to overlap (0.5 means 50% of the score comes from model, 50% from playlist overlap)
        overlap_weight = 0.3  # You can adjust this
        
        print(f"\nAdjusting scores with overlap weight of {overlap_weight:.1f}")
        
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
        
        print("\nTop 20 artists after adjustment:")
        print(ranked_artists[['Adjusted_Rank', 'Artist', 'Predicted_Score', 
                             'In_My_Playlist', 'Adjusted_Score']].head(20))
    
    return ranked_artists