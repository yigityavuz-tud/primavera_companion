from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import os
import uuid
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import time
import gc  # Garbage collector

# Import only what is needed to reduce memory usage
from src.data_processing import load_and_explore_data, preprocess_playlist_data, analyze_genres
from src.modeling import train_and_evaluate_models, predict_and_rank_artists, analyze_artist_overlap

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'primavera-companion-secret-key')

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
RESULT_FOLDER = os.path.join(os.getcwd(), 'results')
ALLOWED_EXTENSIONS = {'csv'}
MAX_ARTISTS = 100  # Limit number of artists to process

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Find Primavera CSV file
PRIMAVERA_CSV = None
possible_locations = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'primavera_25.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'primavera_25.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'primavera.csv'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'primavera.csv'),
    os.path.join(os.getcwd(), 'data', 'primavera_25.csv'),
    os.path.join(os.getcwd(), 'data', 'primavera.csv'),
    os.path.join(os.getcwd(), 'primavera_25.csv'),
    os.path.join(os.getcwd(), 'primavera.csv')
]

for location in possible_locations:
    if os.path.exists(location):
        PRIMAVERA_CSV = location
        logger.info(f"Found Primavera CSV at: {PRIMAVERA_CSV}")
        break

if not PRIMAVERA_CSV:
    logger.warning(f"Warning: Primavera CSV file not found. Checked these locations: {possible_locations}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if primavera file exists
    if not PRIMAVERA_CSV:
        logger.error("Primavera Sound lineup data not found")
        flash('Primavera Sound lineup data not found. Please contact the administrator.', 'error')
        return redirect(url_for('index'))
    
    logger.info(f"Processing file upload. Primavera data at: {PRIMAVERA_CSV}")
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Create a unique session ID
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Create a unique folder for this session
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(session_folder, filename)
        file.save(file_path)
        
        # Create a result folder for this session
        result_folder = os.path.join(app.config['RESULT_FOLDER'], session_id)
        os.makedirs(result_folder, exist_ok=True)
        
        # Store file path in session
        session['my_playlist_path'] = file_path
        session['result_folder'] = result_folder
        
        # Start a background task to process the file
        # For Render, we'll use a synchronous approach but with optimizations
        return redirect(url_for('process'))
    
    flash('Invalid file type. Please upload a CSV file.', 'error')
    return redirect(url_for('index'))

@app.route('/process')
def process():
    # Check if we have the necessary data in the session
    if 'my_playlist_path' not in session or 'result_folder' not in session:
        flash('Session data lost. Please upload your file again.', 'error')
        return redirect(url_for('index'))
    
    my_playlist_path = session['my_playlist_path']
    result_folder = session['result_folder']
    
    # Set up processing message page
    processing_msg = """
    <html>
    <head>
        <title>Processing Your Playlist</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #e63946; }
            .container { max-width: 600px; margin: 0 auto; text-align: center; }
            .spinner { 
                border: 6px solid #f3f3f3;
                border-top: 6px solid #e63946;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 2s linear infinite;
                margin: 30px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .message { margin-top: 20px; font-size: 18px; }
            .note { margin-top: 40px; font-size: 14px; color: #666; }
        </style>
        <meta http-equiv="refresh" content="3;url=/processing_status">
    </head>
    <body>
        <div class="container">
            <h1>Processing Your Playlist</h1>
            <div class="spinner"></div>
            <div class="message">Your playlist is being analyzed to generate recommendations...</div>
            <div class="note">This may take up to 2 minutes. You'll be redirected automatically when completed.</div>
        </div>
    </body>
    </html>
    """
    
    # Start processing in the background
    try:
        # Set a processing flag in the session
        session['processing'] = True
        session['started_at'] = time.time()
        
        return processing_msg
    
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}")
        flash(f'Error processing your playlist: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/processing_status')
def processing_status():
    if 'processing' not in session or not session['processing']:
        return redirect(url_for('index'))
    
    # If we haven't started the processing yet
    if 'started_processing' not in session:
        session['started_processing'] = True
        
        try:
            # Process the data
            process_playlist()
            
            # Clear processing flags
            session['processing'] = False
            
            # Redirect to download
            if 'error' in session and session['error']:
                flash(session['error'], 'error')
                return redirect(url_for('index'))
            else:
                return redirect(url_for('download'))
            
        except Exception as e:
            logger.error(f"Error in processing: {str(e)}")
            session['processing'] = False
            session['error'] = f"Error processing your playlist: {str(e)}"
            flash(session['error'], 'error')
            return redirect(url_for('index'))
    else:
        # Still processing, show the waiting page
        processing_time = time.time() - session.get('started_at', time.time())
        
        # If it's taking too long (more than 2 minutes), assume something went wrong
        if processing_time > 120:
            session['processing'] = False
            flash('Processing is taking longer than expected. Please try again with a smaller playlist.', 'error')
            return redirect(url_for('index'))
        
        processing_msg = f"""
        <html>
        <head>
            <title>Processing Your Playlist</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #e63946; }}
                .container {{ max-width: 600px; margin: 0 auto; text-align: center; }}
                .spinner {{ 
                    border: 6px solid #f3f3f3;
                    border-top: 6px solid #e63946;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 2s linear infinite;
                    margin: 30px auto;
                }}
                @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
                .message {{ margin-top: 20px; font-size: 18px; }}
                .note {{ margin-top: 40px; font-size: 14px; color: #666; }}
            </style>
            <meta http-equiv="refresh" content="3;url=/processing_status">
        </head>
        <body>
            <div class="container">
                <h1>Processing Your Playlist</h1>
                <div class="spinner"></div>
                <div class="message">Processing for {processing_time:.0f} seconds...</div>
                <div class="note">This may take up to 2 minutes. You'll be redirected automatically when completed.</div>
            </div>
        </body>
        </html>
        """
        return processing_msg

def process_playlist():
    """Process the playlist and generate recommendations"""
    my_playlist_path = session['my_playlist_path']
    result_folder = session['result_folder']
    
    # Load the data with limited number of rows to speed up processing
    try:
        # Run the recommendation pipeline with optimizations
        logger.info("Loading playlist data...")
        
        # Read only necessary columns to reduce memory usage
        my_playlist = pd.read_csv(
            my_playlist_path, 
            nrows=10000  # Limit number of rows to avoid timeout
        )
        
        # Load the primavera playlist
        primavera_playlist = pd.read_csv(PRIMAVERA_CSV)
        
        # Basic summary
        n_tracks = len(my_playlist)
        n_primavera_tracks = len(primavera_playlist)
        
        logger.info(f"Loaded {n_tracks} tracks from user playlist and {n_primavera_tracks} from Primavera")
        
        # Get top artists by track count to limit processing
        artist_counts = my_playlist['Artist Name(s)'].str.split(', ').explode().value_counts()
        top_artists = artist_counts.head(MAX_ARTISTS).index.tolist()
        
        # Filter the playlist to only include top artists
        my_playlist_filtered = my_playlist[
            my_playlist['Artist Name(s)'].str.split(', ').apply(
                lambda artists: any(artist in top_artists for artist in artists)
            )
        ]
        
        logger.info(f"Filtered to {len(my_playlist_filtered)} tracks from top {MAX_ARTISTS} artists")
        
        # Analyze genres and find shared ones
        top_shared_genres = analyze_genres(my_playlist_filtered, primavera_playlist)
        
        # Free up memory
        del my_playlist
        gc.collect()
        
        # Process both datasets
        train_data = preprocess_playlist_data(
            my_playlist_filtered, 
            is_training=True, 
            shared_genres=top_shared_genres
        )
        
        # Free up memory
        del my_playlist_filtered
        gc.collect()
        
        test_data = preprocess_playlist_data(
            primavera_playlist, 
            is_training=False, 
            shared_genres=top_shared_genres,
            min_artist_frequency=2  # Lower threshold to include more artists
        )
        
        # Free up memory
        del primavera_playlist
        gc.collect()
        
        # Train and evaluate models
        logger.info("Training models...")
        model_results = train_and_evaluate_models(train_data)
        
        # Predict and rank artists
        logger.info("Ranking artists...")
        ranked_artists, _ = predict_and_rank_artists(test_data, model_results)
        
        # Analyze artist overlap
        # This function is causing the timeout error
        # Simplify it to just add a column indicating overlap
        # Instead of using the original function
        
        # Create a set of artists in the user's playlist for faster lookup
        user_artists = set(train_data['Artist'].unique())
        
        # Check which Primavera artists are in the user's playlist
        ranked_artists['In_My_Playlist'] = ranked_artists['Artist'].apply(
            lambda x: 1 if x in user_artists else 0
        )
        
        # Add a small boost to artists in the playlist
        if 'In_My_Playlist' in ranked_artists.columns:
            # Find maximum predicted score for normalization
            max_score = ranked_artists['Predicted_Score'].max()
            
            # Apply weight to overlap (0.3 means 30% of the score comes from playlist overlap)
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
        
        # Save results to CSV for download
        logger.info("Saving results...")
        csv_path = os.path.join(result_folder, 'primavera_recommendations.csv')
        ranked_artists.to_csv(csv_path, index=False)
        
        # Store results path in session
        session['results_csv_path'] = csv_path
        session['processing_complete'] = True
        
        logger.info("Processing complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error processing playlist: {str(e)}")
        raise e

@app.route('/download')
def download():
    if 'results_csv_path' not in session:
        flash('Download not available. Please process your file again.', 'error')
        return redirect(url_for('index'))
    
    results_csv_path = session['results_csv_path']
    
    if not os.path.exists(results_csv_path):
        flash('Results file not found. Please process your file again.', 'error')
        return redirect(url_for('index'))
    
    # Success message page that auto-triggers the download
    success_msg = """
    <html>
    <head>
        <title>Download Recommendations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #e63946; }
            .container { max-width: 600px; margin: 0 auto; text-align: center; }
            .success-icon {
                color: #2ecc71;
                font-size: 64px;
                margin: 20px 0;
            }
            .message { margin: 20px 0; font-size: 18px; }
            .download-btn {
                display: inline-block;
                background-color: #e63946;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 20px;
                font-weight: bold;
            }
            .home-link {
                display: block;
                margin-top: 40px;
                color: #457b9d;
                text-decoration: none;
            }
            .home-link:hover {
                text-decoration: underline;
            }
        </style>
        <script>
            // Automatically trigger download after 1 second
            window.onload = function() {
                setTimeout(function() {
                    document.getElementById('download-link').click();
                }, 1000);
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Your Recommendations Are Ready!</h1>
            <div class="success-icon">✓</div>
            <div class="message">
                Your personalized Primavera Sound recommendations have been generated successfully.
                Your download should start automatically.
            </div>
            <a id="download-link" href="/download-file" class="download-btn">Download Recommendations</a>
            <a href="/" class="home-link">Return to Home</a>
        </div>
    </body>
    </html>
    """
    
    return success_msg

@app.route('/download-file')
def download_file():
    if 'results_csv_path' not in session:
        flash('Download not available. Please process your file again.', 'error')
        return redirect(url_for('index'))
    
    results_csv_path = session['results_csv_path']
    
    if not os.path.exists(results_csv_path):
        flash('Results file not found. Please process your file again.', 'error')
        return redirect(url_for('index'))
    
    return send_file(
        results_csv_path, 
        as_attachment=True, 
        download_name='primavera_recommendations.csv',
        mimetype='text/csv'
    )


@app.errorhandler(500)
def internal_server_error(e):
    return """
    <html>
    <head>
        <title>Processing Error</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #e63946; }
            .container { max-width: 600px; margin: 0 auto; text-align: center; }
            .error-icon {
                color: #e74c3c;
                font-size: 64px;
                margin: 20px 0;
            }
            .message { margin: 20px 0; font-size: 18px; }
            .suggestion { margin: 20px 0; font-size: 16px; color: #666; }
            .home-btn {
                display: inline-block;
                background-color: #e63946;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 20px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Processing Error</h1>
            <div class="error-icon">⚠</div>
            <div class="message">
                Sorry, we encountered an error while processing your playlist.
            </div>
            <div class="suggestion">
                Your playlist might be too large. Try with a smaller playlist (under 500 tracks).
            </div>
            <a href="/" class="home-btn">Try Again</a>
        </div>
    </body>
    </html>
    """, 500

@app.errorhandler(502)
def bad_gateway_error(e):
    return """
    <html>
    <head>
        <title>Processing Timeout</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #e63946; }
            .container { max-width: 600px; margin: 0 auto; text-align: center; }
            .error-icon {
                color: #e74c3c;
                font-size: 64px;
                margin: 20px 0;
            }
            .message { margin: 20px 0; font-size: 18px; }
            .suggestion { margin: 20px 0; font-size: 16px; color: #666; }
            .home-btn {
                display: inline-block;
                background-color: #e63946;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 20px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Processing Timeout</h1>
            <div class="error-icon">⏱</div>
            <div class="message">
                Sorry, processing your playlist took too long and timed out.
            </div>
            <div class="suggestion">
                Please try again with a smaller playlist (under 500 tracks)
                or select a playlist with fewer unique artists.
            </div>
            <a href="/" class="home-btn">Try Again</a>
        </div>
    </body>
    </html>
    """, 502

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
