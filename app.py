from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, abort
import os
import uuid
import pandas as pd
import json
import base64
import threading
import time
import shutil
import gc  # Garbage collector
import logging

from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Domain-specific imports
from src.data_processing import load_and_explore_data, preprocess_playlist_data, analyze_genres
from src.modeling import train_and_evaluate_models, predict_and_rank_artists, analyze_artist_overlap
from src.visualization import get_graph_as_base64
from src.utils import create_html_result

# Load environment and enforce SECRET_KEY
load_dotenv()
secret_key = os.getenv('SECRET_KEY')
if not secret_key:
    raise RuntimeError('SECRET_KEY environment variable not set')

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

# Configuration constants
PRIMAVERA_CSV = 'data/primavera_25.csv'
UPLOAD_ROOT = './uploads'
RESULT_ROOT = './results'
MAX_ROWS = 4000  # maximum allowed rows in upload
MAX_ARTISTS = 100

# Ensure necessary directories exist
os.makedirs(UPLOAD_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)

# Logging setup
defaulthandler = logging.StreamHandler()
defaulthandler.setLevel(logging.INFO)
logging.getLogger().addHandler(defaulthandler)
logger = logging.getLogger(__name__)

# Background cleanup thread: remove session folders older than 10 minutes

def cleanup_old_folders():
    while True:
        now = time.time()
        for root in (UPLOAD_ROOT, RESULT_ROOT):
            for folder in os.listdir(root):
                path = os.path.join(root, folder)
                if os.path.isdir(path) and now - os.path.getmtime(path) > 600:
                    try:
                        shutil.rmtree(path)
                        logger.info(f"Cleaned up folder {path}")
                    except Exception as e:
                        logger.error(f"Error cleaning up folder {path}: {e}")
        time.sleep(60)

threading.Thread(target=cleanup_old_folders, daemon=True).start()

# Helpers
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Ensure festival data exists
    if not os.path.exists(PRIMAVERA_CSV):
        flash('Festival lineup data not found.', 'error')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file part.', 'error')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file.', 'error')
        return redirect(url_for('index'))
    if not allowed_file(file.filename):
        flash('Invalid file type. Only CSV allowed.', 'error')
        return redirect(url_for('index'))

    # Enforce row limit without loading entire file
    from io import TextIOWrapper
    file.stream.seek(0)
    wrapper = TextIOWrapper(file.stream, encoding='utf-8')
    row_count = sum(1 for _ in wrapper) - 1  # discount header
    file.stream.seek(0)
    if row_count > MAX_ROWS:
        flash(f'Uploaded file has {row_count} rows, exceeding max of {MAX_ROWS}.', 'error')
        return redirect(url_for('index'))

    # Save upload
    uid = uuid.uuid4().hex
    upload_folder = os.path.join(UPLOAD_ROOT, uid)
    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    # Prepare result folder and session
    result_folder = os.path.join(RESULT_ROOT, uid)
    os.makedirs(result_folder, exist_ok=True)
    session['my_playlist_path'] = file_path
    session['result_folder'] = result_folder

    return redirect(url_for('process'))

@app.route('/process')
def process():
    if 'my_playlist_path' not in session or 'result_folder' not in session:
        flash('Session data lost. Please upload your file again.', 'error')
        return redirect(url_for('index'))
    my_path = session['my_playlist_path']
    result_folder = session['result_folder']

    try:
        # 1. Load & explore
        my_df, primavera_df = load_and_explore_data(
            my_playlist_path=my_path,
            primavera_playlist_path=PRIMAVERA_CSV
        )

        # 2. Analyze genres
        shared_genres = analyze_genres(my_df, primavera_df)

        # 3. Preprocess for training/testing
        train_data = preprocess_playlist_data(
            my_df, is_training=True, shared_genres=shared_genres
        )
        del my_df
        gc.collect()

        test_data = preprocess_playlist_data(
            primavera_df, is_training=False, shared_genres=shared_genres,
            min_artist_frequency=5
        )
        del primavera_df
        gc.collect()

        # 4. Train & rank
        model_results = train_and_evaluate_models(train_data)
        ranked = predict_and_rank_artists(model_results, test_data)
        ranked = analyze_artist_overlap(ranked, train_data, test_data)

        # 5. Visualization: save chart to disk
        chart_b64 = get_graph_as_base64(ranked, top_n=30)
        chart_path = os.path.join(result_folder, 'chart.png')
        # strip any data URI prefix
        b64_content = chart_b64.split(',')[-1]
        with open(chart_path, 'wb') as cf:
            cf.write(base64.b64decode(b64_content))

        # 6. Save JSON results
        json_path = os.path.join(result_folder, 'ranked_artists.json')
        with open(json_path, 'w') as jf:
            json.dump({
                'artists': ranked.to_dict(orient='records'),
                'timestamp': pd.Timestamp.now().isoformat()
            }, jf, indent=2)

        # 7. Create standalone HTML
        create_html_result(ranked, output_path=os.path.join(result_folder, 'recommendations.html'))

        return redirect(url_for('results'))
    except Exception as e:
        flash(f'Error processing your playlist: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'result_folder' not in session:
        flash('Processing results not found. Please upload your file again.', 'error')
        return redirect(url_for('index'))
    result_folder = session['result_folder']

    # Load JSON results
    json_path = os.path.join(result_folder, 'ranked_artists.json')
    if not os.path.exists(json_path):
        flash('Results not available. Please try again.', 'error')
        return redirect(url_for('index'))
    with open(json_path) as f:
        data = json.load(f)
    top_artists = data['artists'][:50]
    timestamp = data['timestamp']

    # Load chart for display
    chart_file = os.path.join(result_folder, 'chart.png')
    if os.path.exists(chart_file):
        with open(chart_file, 'rb') as cf:
            chart_data = base64.b64encode(cf.read()).decode('utf-8')
        chart_data = f'data:image/png;base64,{chart_data}'
    else:
        chart_data = None

    return render_template('results.html', chart_data=chart_data, artists=top_artists, timestamp=timestamp)

@app.route('/download')
def download():
    if 'result_folder' not in session:
        flash('Download not available. Please process your file again.', 'error')
        return redirect(url_for('index'))
    html_file = os.path.join(session['result_folder'], 'recommendations.html')
    if not os.path.exists(html_file):
        flash('Download not available. Please try again.', 'error')
        return redirect(url_for('index'))
    return send_file(html_file, as_attachment=True)

# Retain existing 502 handler and main entrypoint
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
            .error-icon { color: #e74c3c; font-size: 64px; margin: 20px 0; }
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
