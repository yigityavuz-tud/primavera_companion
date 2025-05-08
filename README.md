# Primavera Companion

A personalized recommendation system for Primavera Sound festival attendees.

![Primavera Companion](static/images/screenshot.png)

## 🎵 About

Primavera Companion is a machine learning-powered web application that helps music fans discover which artists they should see at the Primavera Sound music festival, based on their personal music taste. The app analyzes a user's Spotify playlist data and recommends festival artists who match their preferences.

## ✨ Features

- **AI-Powered Recommendations**: Uses machine learning to analyze musical features, genres, and artist connections
- **User-Friendly Web Interface**: Upload your playlist and get instant recommendations
- **Visual Results**: See your top recommended artists with beautiful data visualizations
- **Downloadable Results**: Save your personalized festival schedule as an HTML file
- **Data Privacy**: Your playlist data is only used for processing and not stored permanently


## 🚀 Quick Start

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/primavera-companion.git
   cd primavera-companion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration:
   ```
   SECRET_KEY=your_secret_key_here
   ```

4. Place the Primavera lineup data in the `data` directory:
   ```
   data/primavera.csv
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Open your browser and go to `http://localhost:5000`

### Using the Command Line Tool

You can also use the command-line interface:

```bash
python main.py --my-playlist "path/to/your/playlist.csv" --primavera-playlist "data/primavera.csv"
```

## 🧠 How It Works

1. **Data Collection**: Users export their Spotify playlist data using [Exportify](https://exportify.net)
2. **Data Processing**: The system processes both the user's playlist and the Primavera Sound lineup
3. **Feature Engineering**: Identifies shared genres and creates numerical features
4. **Model Training**: Trains multiple regression models to predict artist affinity
5. **Artist Ranking**: Ranks Primavera artists based on predicted scores
6. **Results Visualization**: Presents the recommendations visually

## 🛠️ Technical Details

### Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Matplotlib

### Project Structure

```
primavera-companion/
├── app.py                  # Flask web application
├── main.py                 # Command-line interface
├── requirements.txt        # Dependencies
├── src/                    # Source code
│   ├── data_processing.py  # Data preprocessing functions
│   ├── modeling.py         # ML model training and evaluation
│   ├── visualization.py    # Data visualization functions
│   └── utils.py            # Utility functions
├── static/                 # Static web assets
│   └── css/
│       └── style.css       # Custom styles
├── templates/              # HTML templates
│   ├── index.html          # Homepage
│   ├── results.html        # Results page
│   └── about.html          # About page
├── data/                   # Data directory
│   └── .gitkeep            # (add primavera.csv here)
├── uploads/                # User uploads (temporary)
└── results/                # Generated results
```

## 🎭 Deploying to Production

### Deploying to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Set the build command:
   ```
   pip install -r requirements.txt
   ```
4. Set the start command:
   ```
   gunicorn app:app
   ```
5. Add environment variables:
   ```
   SECRET_KEY=your_secret_key_here
   ```
6. Deploy!

## 🔧 Customization

- Adjust model parameters in `src/modeling.py`
- Modify the UI in `templates/` and `static/css/style.css`
- Update the Primavera lineup data annually in `data/primavera.csv`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👋 Contact

- GitHub: [@yigityavuz_tud](https://github.com/yigityavuz_tud)
- LinkedIn: [Yiğit Yavuz](https://linkedin.com/in/yigit-yavuz)

Feel free to reach out if you have any questions or feedback!