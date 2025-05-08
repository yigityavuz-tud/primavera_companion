# Primavera Companion

A personalized recommendation system for Primavera Sound festival attendees.

## 🎵 About

Primavera Companion is a machine learning-powered tool that helps music fans discover which artists they should see at the Primavera Sound music festival, based on their personal music taste. The app analyzes a user's Spotify playlist data and recommends festival artists who match their preferences.

## ✨ Features

- **AI-Powered Recommendations**: Uses machine learning to analyze musical features, genres, and artist connections
- **Simple Web Interface**: Upload your playlist and get a downloadable CSV of recommendations
- **CSV Output**: Get a ranked list of recommended Primavera artists that matches your music taste
- **Data Privacy**: Your playlist data is only used for processing and not stored permanently

## 🚀 Quick Start

### Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/primavera-companion.git -b webapp
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

5. Run the web application:
   ```bash
   python app.py
   ```

6. Open your browser and go to `http://localhost:5000`

### Using the Command Line Tool

You can also use the command-line interface:

```bash
python main.py --my-playlist "path/to/your/playlist.csv" --primavera-playlist "data/primavera.csv" --output "recommendations.csv"
```

## 🧠 How It Works

1. **Data Collection**: Users export their Spotify playlist data using [Exportify](https://exportify.net)
2. **Data Processing**: The system processes both the user's playlist and the Primavera Sound lineup
3. **Feature Engineering**: Identifies shared genres and creates numerical features
4. **Model Training**: Trains multiple regression models to predict artist affinity
5. **Artist Ranking**: Ranks Primavera artists based on predicted scores
6. **CSV Output**: Provides the recommendations as a downloadable CSV file

## 🛠️ Technical Details

### Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Web Framework**: Flask

### Project Structure

```
primavera-companion/
├── app.py                  # Flask web application
├── main.py                 # Command-line interface
├── requirements.txt        # Dependencies
├── src/                    # Source code
│   ├── data_processing.py  # Data preprocessing functions
│   ├── modeling.py         # ML model training and evaluation
│   └── utils.py            # Utility functions
├── templates/              # HTML templates
│   └── index.html          # Homepage
├── data/                   # Data directory
│   └── .gitkeep            # (add primavera.csv here)
├── uploads/                # User uploads (temporary)
└── results/                # Generated results
```

## 🔧 Customization

- Adjust model parameters in `src/modeling.py`
- Update the Primavera lineup data annually in `data/primavera.csv`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👋 Contact

- GitHub: [@yigityavuz_tud](https://github.com/yigityavuz_tud)
- LinkedIn: [Yiğit Yavuz](https://linkedin.com/in/yigit-yavuz)

Feel free to reach out if you have any questions or feedback!