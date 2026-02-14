# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import base64
import time

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 0.2rem;
    }
    .movie-stats {
        font-size: 0.9rem;
        color: #888;
    }
    .similarity-score {
        font-size: 0.9rem;
        color: #00FF00;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #FF4B4B;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# Function to load data with caching
@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        # Load the movie data
        df = pd.read_csv('movie_data_for_app.csv')
        
        # Load the saved models
        with open('sigmoid_kernel.pkl', 'rb') as f:
            sigmoid_kernel = pickle.load(f)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        
        return df, sigmoid_kernel, tfidf
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.info("Please ensure all required files are in the correct directory.")
        return None, None, None

# Function to get movie poster from TMDB API
@st.cache_data
def get_movie_poster(movie_title):
    """Fetch movie poster from TMDB API"""
    try:
        # Note: You'll need to add your TMDB API key
        api_key = "YOUR_TMDB_API_KEY"  # Replace with your API key
        base_url = "https://api.themoviedb.org/3/search/movie"
        
        params = {
            'api_key': api_key,
            'query': movie_title
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                poster_path = results[0].get('poster_path')
                if poster_path:
                    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                    return poster_url
    except:
        pass
    return None

# Function to get recommendations
def get_recommendations(title, df, sigmoid_kernel, indices, num_recommendations=10):
    """Get movie recommendations based on sigmoid kernel similarity"""
    idx = indices[title]
    sig_scores = list(enumerate(sigmoid_kernel[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sig_scores]
    similarities = [i[1] for i in sig_scores]
    
    recommendations = df.iloc[movie_indices].copy()
    recommendations['similarity_score'] = similarities
    
    return recommendations

# Function to get weighted rating recommendations
def get_weighted_recommendations(df, min_votes_threshold=1000, num_recommendations=10):
    """Get recommendations based on weighted rating (IMDB formula)"""
    v = df['vote_count']
    R = df['vote_average']
    C = df['vote_average'].mean()
    m = min_votes_threshold
    
    df['weighted_score'] = (v/(v+m) * R) + (m/(v+m) * C)
    return df.nlargest(num_recommendations, 'weighted_score')

# Function to get popularity-based recommendations
def get_popularity_recommendations(df, num_recommendations=10):
    """Get recommendations based on popularity"""
    return df.nlargest(num_recommendations, 'popularity')

# Function to get hybrid recommendations
def get_hybrid_recommendations(df, min_votes_threshold=1000, num_recommendations=10):
    """Get recommendations combining weighted rating and popularity"""
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    
    v = df['vote_count']
    R = df['vote_average']
    C = df['vote_average'].mean()
    m = min_votes_threshold
    
    df['weighted_score'] = (v/(v+m) * R) + (m/(v+m) * C)
    
    scaled_features = scaler.fit_transform(df[['weighted_score', 'popularity']])
    df['hybrid_score'] = scaled_features.mean(axis=1)
    
    return df.nlargest(num_recommendations, 'hybrid_score')

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover your next favorite movie using advanced ML techniques</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading movie database...'):
        df, sigmoid_kernel, tfidf = load_data()
    
    if df is None:
        st.stop()
    
    # Create indices for lookup
    indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.themoviedb.org/assets/2/v4/logos/v2/blue_square_2-d537fb228cf3ded904ef09b136fe3fec72548ebc1fea3fbbd1ad9e36364db38b.svg", 
                 width=200)
        st.markdown("## 🎯 Navigation")
        app_mode = st.radio(
            "Choose Mode",
            ["🎯 Get Recommendations", "📊 Explore Data", "🔍 Compare Models", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Movies", f"{len(df):,}")
        with col2:
            st.metric("Avg Rating", f"{df['vote_average'].mean():.2f}")
        
        st.markdown("---")
        st.markdown("### 🎨 Customize")
        num_recommendations = st.slider("Number of recommendations", 5, 20, 10)
        show_posters = st.checkbox("Show movie posters", value=True)
    
    # Main content based on selected mode
    if app_mode == "🎯 Get Recommendations":
        st.markdown("## 🎯 Get Personalized Recommendations")
        
        # Create tabs for different recommendation methods
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Content-Based (TF-IDF)", 
            "⭐ Weighted Rating", 
            "📈 Popularity", 
            "🔄 Hybrid"
        ])
        
        with tab1:
            st.markdown("### Find movies similar to your favorites")
            st.markdown("*Using TF-IDF vectorization + Sigmoid Kernel similarity*")
            
            # Movie selection
            movie_titles = df['original_title'].tolist()
            selected_movie = st.selectbox(
                "Select a movie you like:",
                movie_titles,
                key="content_select"
            )
            
            if st.button("Get Recommendations", key="content_btn"):
                with st.spinner("Finding similar movies..."):
                    recommendations = get_recommendations(
                        selected_movie, df, sigmoid_kernel, indices, num_recommendations
                    )
                    st.session_state.recommendations = recommendations
                    st.session_state.selected_movie = selected_movie
            
            if st.session_state.recommendations is not None and st.session_state.selected_movie == selected_movie:
                st.success(f"🎉 Movies similar to **{selected_movie}**:")
                
                # Display recommendations in a grid
                cols = st.columns(3)
                for idx, (_, movie) in enumerate(st.session_state.recommendations.iterrows()):
                    with cols[idx % 3]:
                        with st.container():
                            st.markdown(f"<div class='recommendation-card'>", unsafe_allow_html=True)
                            
                            # Movie poster
                            if show_posters:
                                poster_url = get_movie_poster(movie['original_title'])
                                if poster_url:
                                    st.image(poster_url, use_container_width=True)
                            
                            # Movie info
                            st.markdown(f"**{movie['original_title']}**")
                            st.markdown(f"⭐ {movie['vote_average']:.1f} | 📊 {movie['popularity']:.1f}")
                            st.markdown(f"🎭 {movie['genres'][:50]}..." if len(str(movie['genres'])) > 50 else f"🎭 {movie['genres']}")
                            st.markdown(f"<span class='similarity-score'>Similarity: {movie['similarity_score']:.2%}</span>", 
                                      unsafe_allow_html=True)
                            
                            # Show overview in expander
                            with st.expander("📖 Overview"):
                                st.write(movie['overview'][:200] + "..." if len(str(movie['overview'])) > 200 else movie['overview'])
                            
                            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("### Top Rated Movies")
            st.markdown("*Using IMDB weighted rating formula*")
            
            min_votes = st.slider("Minimum votes threshold", 100, 5000, 1000, step=100, key="weighted_votes")
            
            weighted_recs = get_weighted_recommendations(df, min_votes, num_recommendations)
            
            # Display as table
            st.dataframe(
                weighted_recs[['original_title', 'vote_average', 'vote_count', 'weighted_score', 'popularity']]
                .style.format({
                    'vote_average': '{:.1f}',
                    'weighted_score': '{:.3f}',
                    'popularity': '{:.1f}'
                }),
                use_container_width=True
            )
            
            # Bar chart
            fig = px.bar(
                weighted_recs.head(10),
                x='weighted_score',
                y='original_title',
                orientation='h',
                title='Top 10 Movies by Weighted Rating',
                color='weighted_score',
                color_continuous_scale='viridis'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Most Popular Movies")
            st.markdown("*Based on TMDB popularity score*")
            
            popular_recs = get_popularity_recommendations(df, num_recommendations)
            
            # Display as table
            st.dataframe(
                popular_recs[['original_title', 'popularity', 'vote_average', 'vote_count']]
                .style.format({'popularity': '{:.1f}', 'vote_average': '{:.1f}'}),
                use_container_width=True
            )
            
            # Bar chart
            fig = px.bar(
                popular_recs.head(10),
                x='popularity',
                y='original_title',
                orientation='h',
                title='Top 10 Most Popular Movies',
                color='popularity',
                color_continuous_scale='plasma'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### Hybrid Recommendations")
            st.markdown("*Combining weighted rating and popularity (scaled)*")
            
            min_votes_hybrid = st.slider("Minimum votes threshold", 100, 5000, 1000, step=100, key="hybrid_votes")
            
            hybrid_recs = get_hybrid_recommendations(df, min_votes_hybrid, num_recommendations)
            
            # Display as table
            st.dataframe(
                hybrid_recs[['original_title', 'vote_average', 'popularity', 'weighted_score', 'hybrid_score']]
                .style.format({
                    'vote_average': '{:.1f}',
                    'popularity': '{:.1f}',
                    'weighted_score': '{:.3f}',
                    'hybrid_score': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Scatter plot
            fig = px.scatter(
                hybrid_recs,
                x='weighted_score',
                y='popularity',
                size='vote_count',
                hover_data=['original_title'],
                title='Hybrid Score Distribution',
                color='hybrid_score',
                color_continuous_scale='rainbow'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "📊 Explore Data":
        st.markdown("## 📊 Explore Movie Database")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_rating = st.slider("Minimum rating", 0.0, 10.0, 5.0, 0.1)
        with col2:
            min_votes = st.number_input("Minimum votes", min_value=0, value=100)
        with col3:
            search = st.text_input("Search movies", "")
        
        # Filter data
        filtered_df = df[(df['vote_average'] >= min_rating) & (df['vote_count'] >= min_votes)]
        if search:
            filtered_df = filtered_df[filtered_df['original_title'].str.contains(search, case=False)]
        
        st.markdown(f"### Showing {len(filtered_df)} movies")
        
        # Display with pagination
        page_size = 20
        total_pages = max(1, len(filtered_df) // page_size + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        st.dataframe(
            filtered_df.iloc[start_idx:end_idx][
                ['original_title', 'vote_average', 'vote_count', 'popularity', 'genres', 'release_date']
            ],
            use_container_width=True
        )
        
        # Visualizations
        st.markdown("### 📈 Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            fig = px.histogram(
                df, 
                x='vote_average', 
                nbins=50,
                title='Distribution of Movie Ratings',
                labels={'vote_average': 'Rating'},
                color_discrete_sequence=['#FF4B4B']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Popularity distribution
            fig = px.scatter(
                df.sample(min(1000, len(df))),
                x='vote_average',
                y='popularity',
                size='vote_count',
                hover_data=['original_title'],
                title='Rating vs Popularity',
                color='vote_average',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif app_mode == "🔍 Compare Models":
        st.markdown("## 🔍 Compare Recommendation Models")
        
        movie_titles = df['original_title'].tolist()
        test_movie = st.selectbox("Select a test movie:", movie_titles, key="compare_select")
        
        if st.button("Compare All Models", key="compare_btn"):
            with st.spinner("Generating comparisons..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎯 Content-Based")
                    content_recs = get_recommendations(test_movie, df, sigmoid_kernel, indices, 5)
                    for _, movie in content_recs.iterrows():
                        st.markdown(f"- **{movie['original_title']}** (similarity: {movie['similarity_score']:.2%})")
                
                with col2:
                    st.markdown("### ⭐ Weighted Rating")
                    weighted_recs = get_weighted_recommendations(df, 1000, 5)
                    for _, movie in weighted_recs.iterrows():
                        st.markdown(f"- **{movie['original_title']}** (score: {movie['weighted_score']:.3f})")
                
                st.markdown("---")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("### 📈 Popularity")
                    popular_recs = get_popularity_recommendations(df, 5)
                    for _, movie in popular_recs.iterrows():
                        st.markdown(f"- **{movie['original_title']}** (popularity: {movie['popularity']:.1f})")
                
                with col4:
                    st.markdown("### 🔄 Hybrid")
                    hybrid_recs = get_hybrid_recommendations(df, 1000, 5)
                    for _, movie in hybrid_recs.iterrows():
                        st.markdown(f"- **{movie['original_title']}** (hybrid: {movie['hybrid_score']:.3f})")
    
    else:  # About
        st.markdown("## ℹ️ About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        This Movie Recommendation System uses multiple approaches to help you discover new movies:
        
        1. **Content-Based Filtering** using TF-IDF vectorization and Sigmoid Kernel similarity
        2. **Weighted Rating** based on the IMDB formula
        3. **Popularity-Based** recommendations
        4. **Hybrid Model** combining multiple metrics
        
        ### 🛠️ Technical Implementation
        - **Data Source**: TMDB 5000 Movie Dataset
        - **Text Processing**: TF-IDF Vectorization
        - **Similarity Metric**: Sigmoid Kernel
        - **Frontend**: Streamlit
        - **Visualizations**: Plotly
        
        ### 📊 Files Used
        - `movie_data_for_app.csv` - Processed movie data
        - `sigmoid_kernel.pkl` - Pre-computed similarity matrix
        - `tfidf_vectorizer.pkl` - Trained TF-IDF vectorizer
        
        ### 🎓 How It Works
        
        **TF-IDF Vectorization** converts movie descriptions, keywords, and genres into numerical vectors,
        weighting words based on their importance. The **Sigmoid Kernel** then calculates similarity between
        these vectors to find movies with similar content patterns.
        
        ### 👨‍💻 Created By
        A movie enthusiast who believes in the power of data to help people discover great films!
        """)
        
        # Show sample of the data
        st.markdown("### 📋 Sample Data")
        st.dataframe(df.head(10)[['original_title', 'vote_average', 'popularity', 'genres']])
    
    # Footer
    st.markdown('<div class="footer">Made with ❤️ using Streamlit and TMDB data</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()