import os
import json
import pandas as pd

def save_results(ranked_artists, output_dir="./results"):
    """Save the ranked artists to CSV and JSON files"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "ranked_artists.csv")
    ranked_artists.to_csv(csv_path, index=False)
    
    # Save to JSON (more convenient for web display)
    json_path = os.path.join(output_dir, "ranked_artists.json")
    
    # Convert to a format suitable for web display
    json_data = {
        "artists": ranked_artists.to_dict(orient="records"),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Results saved to {csv_path} and {json_path}")
    
    return csv_path, json_path


def create_html_result(ranked_artists, output_path="./results/recommendations.html"):
    """Create a standalone HTML file with the recommendations"""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get top 50 artists
    top_artists = ranked_artists.head(50).copy()
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Primavera Sound Recommendations</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }
            h1, h2 {
                color: #e63946;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #e63946;
                padding-bottom: 10px;
            }
            .artist-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .artist-card {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .artist-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            .rank {
                font-size: 1.5em;
                font-weight: bold;
                color: #e63946;
                margin-right: 10px;
            }
            .artist-name {
                font-weight: bold;
                font-size: 1.2em;
                margin-bottom: 5px;
            }
            .score {
                color: #777;
                font-size: 0.9em;
            }
            .in-playlist {
                background-color: #f8f9fa;
                border-left: 4px solid #e63946;
            }
            .footer {
                margin-top: 40px;
                text-align: center;
                font-size: 0.8em;
                color: #777;
            }
            @media (max-width: 600px) {
                .artist-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Primavera Sound Festival</h1>
            <h2>Your Personalized Artist Recommendations</h2>
            <p>Based on your music taste</p>
        </div>
        
        <p>Here are the top artists at Primavera Sound that match your music taste:</p>
        
        <div class="artist-container">
    """
    
    # Add artist cards
    for _, row in top_artists.iterrows():
        in_playlist_class = " in-playlist" if 'In_My_Playlist' in row and row['In_My_Playlist'] == 1 else ""
        rank = row.get('Adjusted_Rank', row['Rank'])
        score = row.get('Adjusted_Score', row['Predicted_Score'])
        
        html_content += f"""
            <div class="artist-card{in_playlist_class}">
                <div class="artist-name"><span class="rank">{int(rank)}</span>{row['Artist']}</div>
                <div class="score">Match score: {score:.2f}</div>
                {f'<div class="in-your-playlist">✓ In your playlist</div>' if in_playlist_class else ''}
            </div>
        """
    
    # Close HTML content
    html_content += """
        </div>
        
        <div class="footer">
            <p>Generated by Primavera Companion - Your Festival AI Assistant</p>
            <p>Created with ❤️ for music lovers</p>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML result saved to {output_path}")
    
    return output_path