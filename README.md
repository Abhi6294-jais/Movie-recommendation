
# 🎬 Movie Recommendation System

## 🚀 Live Demo

Check out the live application here: [Movie Recommendation App](https://movie-recommendation-8tvtaptosb6.streamlit.app/)

A sophisticated machine learning-based web application that provides personalized movie recommendations. This system utilizes multiple recommendation strategies including Content-Based Filtering, Weighted Ratings, and Hybrid approaches to suggest the best movies to users.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange)

## ✨ Features

- **🎯 Content-Based Filtering**: Suggests movies similar to your favorites based on plot summaries, genres, and keywords using TF-IDF Vectorization and Sigmoid Kernel similarity.
- **⭐ Top Rated (Weighted)**: Discovers highly-rated movies using the IMDB weighted rating formula to ensure a fair ranking between popular and niche films.
- **📈 Popularity-Based**: Shows currently trending and popular movies based on TMDB popularity scores.
- **🔄 Hybrid Recommendations**: A smart combination of weighted ratings and popularity to find the best all-around movies.
- **📊 Interactive Data Exploration**: Search, filter, and visualize the movie database with interactive charts using Plotly.
- **🖼️ Visual Interface**:  Fetches live movie posters and details via the TMDB API (requires API key).

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (TF-IDF, Cosine Similarity)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **API Integration**: Requests (TMDB API)

## 📂 Project Structure

```
├── frontend.py               # Main Streamlit application
├── Recc_system_tmdb.ipynb    # Jupyter notebook for data analysis & model building
├── generate_models.py        # Script to process data and generate model files
├── requirement.txt           # Project dependencies
├── movie_data_for_app.csv    # Processed dataset for the app
├── sigmoid_kernel.pkl        # Pre-computed similarity matrix (Large file)
├── tfidf_vectorizer.pkl      # Trained TF-IDF model
└── tmdb_5000_*.csv           # Raw datasets
```

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Movie_recommendation
   ```

2. **Create a virtual environment (Optional but recommended)**
   ```bash
   python -m venv myenv
   # Windows
   .\myenv\Scripts\activate
   # Mac/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Prepare the Data**
   If the model files (`.pkl`) are missing, you can generate them by running:
   ```bash
   python generate_models.py
   ```
   *Note: Ensure `tmdb_5000_credits.csv` and `tmdb_5000_movies.csv` are in the directory.*

## ▶️ Running the App

Run the Streamlit application:

```bash
streamlit run frontend.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 💾 Dataset

The project uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle.
- `tmdb_5000_movies.csv`: Contains metadata like budget, genre, homepage, id, keywords, original_language, etc.
- `tmdb_5000_credits.csv`: Contains cast and crew information.

## 📝 Notes

- **Git LFS**: The `sigmoid_kernel.pkl` file is large (~180MB). If pushing to GitHub, ensure you have **Git LFS (Large File Storage)** installed and configured.
- **API Key**: To see movie posters, you need to replace `YOUR_TMDB_API_KEY` in `frontend.py` with your actual API key from [The Movie Database](https://www.themoviedb.org/documentation/api).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📜 License

[MIT](https://choosealicense.com/licenses/mit/)
