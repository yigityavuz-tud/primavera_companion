import pandas as pd
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

def plot_artist_distribution(ranked_artists, top_n=30, output_dir="./results"):
    """Create a horizontal bar chart of the top artists and their scores"""
    plt.figure(figsize=(10, 8))
    
    # Select the column to use (adjusted score if available, otherwise predicted score)
    score_col = 'Adjusted_Score' if 'Adjusted_Score' in ranked_artists.columns else 'Predicted_Score'
    rank_col = 'Adjusted_Rank' if 'Adjusted_Rank' in ranked_artists.columns else 'Rank'
    
    # Get top N artists
    top_artists = ranked_artists.sort_values(rank_col).head(top_n)
    
    # Create horizontal bar chart
    bars = plt.barh(
        top_artists['Artist'], 
        top_artists[score_col],
        color=['#e63946' if in_playlist else '#457b9d' 
               for in_playlist in top_artists.get('In_My_Playlist', [0] * len(top_artists))]
    )
    
    # Add labels and title
    plt.xlabel('Match Score')
    plt.title(f'Top {top_n} Recommended Artists for Primavera Sound')
    plt.gca().invert_yaxis()  # To have rank 1 at the top
    
    # Add a legend if we have the in-playlist information
    if 'In_My_Playlist' in top_artists.columns:
        plt.legend(
            [plt.Rectangle((0, 0), 1, 1, color='#e63946'), 
             plt.Rectangle((0, 0), 1, 1, color='#457b9d')],
            ['In Your Playlist', 'New Discovery']
        )
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save figure
    output_path = os.path.join(output_dir, 'artist_recommendations.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Chart saved to {output_path}")
    
    return output_path


def get_graph_as_base64(ranked_artists, top_n=30):
    """Generate a base64 encoded image of the top artists graph for web embedding"""
    plt.figure(figsize=(10, 8))
    
    # Select the column to use (adjusted score if available, otherwise predicted score)
    score_col = 'Adjusted_Score' if 'Adjusted_Score' in ranked_artists.columns else 'Predicted_Score'
    rank_col = 'Adjusted_Rank' if 'Adjusted_Rank' in ranked_artists.columns else 'Rank'
    
    # Get top N artists
    top_artists = ranked_artists.sort_values(rank_col).head(top_n)
    
    # Create horizontal bar chart
    bars = plt.barh(
        top_artists['Artist'], 
        top_artists[score_col],
        color=['#e63946' if in_playlist else '#457b9d' 
               for in_playlist in top_artists.get('In_My_Playlist', [0] * len(top_artists))]
    )
    
    # Add labels and title
    plt.xlabel('Match Score')
    plt.title(f'Top {top_n} Recommended Artists for Primavera Sound')
    plt.gca().invert_yaxis()  # To have rank 1 at the top
    
    # Add a legend if we have the in-playlist information
    if 'In_My_Playlist' in top_artists.columns:
        plt.legend(
            [plt.Rectangle((0, 0), 1, 1, color='#e63946'), 
             plt.Rectangle((0, 0), 1, 1, color='#457b9d')],
            ['In Your Playlist', 'New Discovery']
        )
    
    plt.tight_layout()
    
    # Save to a BytesIO object
    img_bytes = BytesIO()
    plt.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
    img_bytes.seek(0)
    
    # Convert to base64
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"


def plot_feature_importance(model_results, feature_names, output_dir="./results"):
    """Plot feature importance from the Random Forest model"""
    # Check if Random Forest is in the results
    if 'Random Forest' not in model_results:
        print("Random Forest model not found in results. Skipping feature importance plot.")
        return None
    
    # Get the model
    rf_model = model_results['Random Forest']['model']
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance.head(15)['Feature'], feature_importance.head(15)['Importance'])
    plt.xlabel('Importance')
    plt.title('Top 15 Most Important Features (Random Forest)')
    plt.gca().invert_yaxis()  # To have most important at the top
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save figure
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Feature importance chart saved to {output_path}")
    
    return output_path