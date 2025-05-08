from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import os
import uuid
import pandas as pd
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from src.data_processing import load_and_explore_data, preprocess_playlist_data, analyze_genres
from src.modeling import train_and_evaluate_models, predict_and_rank_artists, analyze_artist_overlap

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'primavera-companion-secret-key')

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
RESULT_FOLDER = os.path.join(os.getcwd(), 'results')
ALLOWED_EXTENSIONS = {'csv'}

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Check if primavera.csv exists in data folder
PRIMAVERA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'primavera_25.csv')
if not os.path.exists(PRIMAVERA_CSV):
    print(f"Warning: Primavera CSV file not found at {PRIMAVERA_CSV}. Please place it there.")
    

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if primavera file exists
    if not os.path.exists(PRIMAVERA_CSV):
        flash('Primavera Sound lineup data not found. Please contact the administrator.', 'error')
        return redirect(url_for('index'))
    
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
    
    try:
        # Run the recommendation pipeline
        my_playlist, primavera_playlist = load_and_explore_data(
            my_playlist_path=my_playlist_path,
            primavera_playlist_path=PRIMAVERA_CSV
        )

        # Analyze genres and find shared ones
        top_shared_genres = analyze_genres(my_playlist, primavera_playlist)

        # Process both datasets
        train_data = preprocess_playlist_data(
            my_playlist, 
            is_training=True, 
            shared_genres=top_shared_genres
        )

        test_data = preprocess_playlist_data(
            primavera_playlist, 
            is_training=False, 
            shared_genres=top_shared_genres,
            min_artist_frequency=5
        )

        # Train and evaluate models
        model_results = train_and_evaluate_models(train_data)

        # Predict and rank artists
        ranked_artists, predicted_test_data = predict_and_rank_artists(test_data, model_results)

        # Analyze artist overlap
        ranked_artists = analyze_artist_overlap(
            ranked_artists, my_playlist, primavera_playlist
        )
        
        # Save results to CSV for download
        csv_path = os.path.join(result_folder, 'primavera_recommendations.csv')
        ranked_artists.to_csv(csv_path, index=False)
        
        # Store results path in session
        session['results_csv_path'] = csv_path
        
        return redirect(url_for('download'))
    
    except Exception as e:
        flash(f'Error processing your playlist: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/download')
def download():
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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
