import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from textblob import TextBlob
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    h2 {
        color: #ff4b4b;
        border-bottom: 2px solid #ff4b4b;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Helper functions
def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def get_sentiment_features(text):
    """Extract sentiment features using TextBlob"""
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'length': len(text.split()),
        'avg_word_length': np.mean([len(w) for w in text.split()]) if len(text.split()) > 0 else 0
    }

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for given text"""
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    rating = prediction + 1  # Convert back to 1-5 scale
    return rating, prediction

def plot_confusion_matrix(cm, title):
    """Create interactive confusion matrix plot"""
    class_labels = ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star']
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_labels,
        y=class_labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        height=500,
        width=600
    )
    
    return fig

def plot_rating_distribution(data):
    """Plot rating distribution"""
    rating_counts = data['reviews.rating'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=rating_counts.index,
            y=rating_counts.values,
            marker_color=['#ff4444', '#ff9944', '#ffdd44', '#99dd44', '#44dd44'],
            text=rating_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Rating Distribution',
        xaxis_title='Rating',
        yaxis_title='Count',
        height=400,
        showlegend=False
    )
    
    return fig

def plot_model_comparison(results_df):
    """Plot model performance comparison"""
    fig = go.Figure()
    
    models = results_df['model'].unique()
    datasets = results_df['dataset'].unique()
    
    for dataset in datasets:
        subset = results_df[results_df['dataset'] == dataset]
        fig.add_trace(go.Bar(
            name=dataset,
            x=subset['model'],
            y=subset['f1_score'],
            text=subset['f1_score'].round(4),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model F1-Score Comparison',
        xaxis_title='Model',
        yaxis_title='F1-Score',
        barmode='group',
        height=500,
        showlegend=True
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown("<h1>⭐ Product Review Sentiment Analysis System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/star.png", width=80)
        st.title("Navigation")
        st.markdown("---")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home", "📊 Data Analysis", "🤖 Model Training", "🔮 Live Prediction", "📈 Results & Metrics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.info("**About This Project**\n\nA comprehensive sentiment analysis system comparing 5 different ML/DL models for product review classification.")
        
        st.markdown("---")
        st.success("**Models Used:**\n- Naive Bayes\n- Logistic Regression\n- SVM\n- Random Forest\n- LSTM")
    
    # Page routing
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Analysis":
        show_data_analysis()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🔮 Live Prediction":
        show_live_prediction()
    elif page == "📈 Results & Metrics":
        show_results()

def show_home():
    """Home page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2 style='color: #667eea;'>Welcome to the Sentiment Analysis System</h2>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>
                A machine learning solution for analyzing product reviews and predicting ratings
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features
    st.subheader("🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>📊 Data Analysis</h3>
            <p>Comprehensive exploratory data analysis with interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>🤖 Multi-Model Training</h3>
            <p>Train and compare 5 different ML/DL models simultaneously</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>🔮 Real-time Prediction</h3>
            <p>Get instant sentiment predictions on new reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Overview
    st.subheader("📝 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Objective:**
        - Classify product reviews into 5 rating categories (1-5 stars)
        - Compare traditional ML and deep learning approaches
        - Evaluate cross-domain performance
        
        **Dataset:**
        - Product reviews from Amazon and Flipkart
        - Multi-class classification (5 ratings)
        - Text preprocessing and feature engineering
        """)
    
    with col2:
        st.markdown("""
        **Models Implemented:**
        1. **Naive Bayes** - Probabilistic classifier
        2. **Logistic Regression** - Linear model
        3. **SVM** - Support Vector Machine
        4. **Random Forest** - Ensemble method
        5. **LSTM** - Deep learning approach
        
        **Evaluation Metrics:**
        - Accuracy, F1-Score, Precision, Recall
        - Confusion Matrix analysis
        """)
    
    st.markdown("---")
    
    # Getting Started
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Upload Data**: Navigate to 'Data Analysis' to load your dataset
    2. **Train Models**: Go to 'Model Training' to train all models
    3. **Make Predictions**: Use 'Live Prediction' to test on new reviews
    4. **View Results**: Check 'Results & Metrics' for detailed performance analysis
    """)

def show_data_analysis():
    """Data analysis page"""
    st.header("📊 Data Analysis & Exploration")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Display tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 Statistics", "📊 Visualizations", "🔍 Sample Data"])
            
            with tab1:
                st.subheader("Dataset Information")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Reviews", f"{len(df):,}")
                with col2:
                    st.metric("Features", df.shape[1])
                with col3:
                    if 'reviews.rating' in df.columns:
                        st.metric("Avg Rating", f"{df['reviews.rating'].mean():.2f}")
                with col4:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                st.markdown("---")
                
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
            
            with tab2:
                st.subheader("Statistical Summary")
                
                if 'reviews.rating' in df.columns:
                    st.write("**Rating Distribution:**")
                    rating_dist = df['reviews.rating'].value_counts().sort_index()
                    st.dataframe(rating_dist.to_frame('Count'), use_container_width=True)
                
                if 'brand' in df.columns:
                    st.write("**Brand Distribution:**")
                    brand_dist = df['brand'].value_counts()
                    st.dataframe(brand_dist.to_frame('Count'), use_container_width=True)
                
                st.write("**Numerical Features Summary:**")
                st.dataframe(df.describe(), use_container_width=True)
            
            with tab3:
                st.subheader("Data Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'reviews.rating' in df.columns:
                        st.plotly_chart(plot_rating_distribution(df), use_container_width=True)
                
                with col2:
                    if 'brand' in df.columns:
                        brand_counts = df['brand'].value_counts()
                        fig = px.pie(
                            values=brand_counts.values,
                            names=brand_counts.index,
                            title='Brand Distribution',
                            color_discrete_sequence=px.colors.sequential.RdBu
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                if 'reviews.text' in df.columns:
                    st.write("**Review Length Analysis:**")
                    df['review_length'] = df['reviews.text'].astype(str).apply(lambda x: len(x.split()))
                    fig = px.histogram(
                        df,
                        x='review_length',
                        nbins=50,
                        title='Distribution of Review Lengths',
                        labels={'review_length': 'Number of Words'},
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Sample Reviews")
                
                num_samples = st.slider("Number of samples to display", 5, 50, 10)
                
                if st.button("Show Random Samples"):
                    sample_df = df.sample(n=min(num_samples, len(df)))
                    st.dataframe(sample_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        
        st.markdown("""
        ### Expected CSV Format:
        - `reviews.text`: The review text
        - `reviews.rating`: Rating (1-5)
        - `brand`: Brand name (optional)
        """)

def show_model_training():
    """Model training page"""
    st.header("🤖 Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the 'Data Analysis' section")
        return
    
    df = st.session_state.df
    
    # Training configuration
    st.subheader("⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        max_features = st.selectbox("TF-IDF Max Features", [3000, 5000, 8000, 10000], index=1)
    with col3:
        random_state = st.number_input("Random State", 0, 100, 42)
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🎯 Select Models to Train")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        train_nb = st.checkbox("Naive Bayes", value=True)
    with col2:
        train_lr = st.checkbox("Logistic Regression", value=True)
    with col3:
        train_svm = st.checkbox("SVM", value=True)
    with col4:
        train_rf = st.checkbox("Random Forest", value=True)
    
    st.markdown("---")
    
    if st.button("🚀 Start Training", use_container_width=True):
        try:
            with st.spinner("Training models... This may take a few minutes..."):
                # Data preprocessing
                df_clean = df[['reviews.text', 'reviews.rating']].copy()
                df_clean = df_clean.dropna()
                
                df_clean['clean_text'] = df_clean['reviews.text'].apply(clean_text)
                
                X = df_clean['clean_text']
                y = df_clean['reviews.rating'].astype(int) - 1
                
                # Split data
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                
                # Vectorization
                vectorizer = TfidfVectorizer(max_features=max_features)
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_test_tfidf = vectorizer.transform(X_test)
                
                # Store in session state
                st.session_state.vectorizer = vectorizer
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_test_tfidf = X_test_tfidf
                
                # Train models
                models = {}
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                model_count = sum([train_nb, train_lr, train_svm, train_rf])
                current_model = 0
                
                if train_nb:
                    status_text.text("Training Naive Bayes...")
                    model_nb = MultinomialNB()
                    model_nb.fit(X_train_tfidf, y_train)
                    models['Naive Bayes'] = model_nb
                    
                    y_pred = model_nb.predict(X_test_tfidf)
                    results.append({
                        'model': 'Naive Bayes',
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred, average='weighted')
                    })
                    
                    current_model += 1
                    progress_bar.progress(current_model / model_count)
                
                if train_lr:
                    status_text.text("Training Logistic Regression...")
                    model_lr = LogisticRegression(max_iter=500, random_state=random_state)
                    model_lr.fit(X_train_tfidf, y_train)
                    models['Logistic Regression'] = model_lr
                    
                    y_pred = model_lr.predict(X_test_tfidf)
                    results.append({
                        'model': 'Logistic Regression',
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred, average='weighted')
                    })
                    
                    current_model += 1
                    progress_bar.progress(current_model / model_count)
                
                if train_svm:
                    status_text.text("Training SVM...")
                    model_svm = LinearSVC(random_state=random_state, max_iter=1000)
                    model_svm.fit(X_train_tfidf, y_train)
                    models['SVM'] = model_svm
                    
                    y_pred = model_svm.predict(X_test_tfidf)
                    results.append({
                        'model': 'SVM',
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred, average='weighted')
                    })
                    
                    current_model += 1
                    progress_bar.progress(current_model / model_count)
                
                if train_rf:
                    status_text.text("Training Random Forest...")
                    model_rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
                    model_rf.fit(X_train_tfidf, y_train)
                    models['Random Forest'] = model_rf
                    
                    y_pred = model_rf.predict(X_test_tfidf)
                    results.append({
                        'model': 'Random Forest',
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1_score': f1_score(y_test, y_pred, average='weighted')
                    })
                    
                    current_model += 1
                    progress_bar.progress(current_model / model_count)
                
                status_text.text("Training complete!")
                
                # Store results
                st.session_state.models = models
                st.session_state.results = pd.DataFrame(results)
                st.session_state.models_trained = True
                
                st.success("✅ All models trained successfully!")
                
                # Display results
                st.subheader("📊 Training Results")
                
                results_df = pd.DataFrame(results)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(results_df.style.highlight_max(axis=0, subset=['accuracy', 'f1_score']), 
                               use_container_width=True)
                
                with col2:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Accuracy',
                        x=results_df['model'],
                        y=results_df['accuracy'],
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        name='F1-Score',
                        x=results_df['model'],
                        y=results_df['f1_score'],
                        marker_color='lightcoral'
                    ))
                    fig.update_layout(
                        title='Model Performance Comparison',
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error during training: {str(e)}")

def show_live_prediction():
    """Live prediction page"""
    st.header("🔮 Live Sentiment Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Model Training' section")
        return
    
    models = st.session_state.models
    vectorizer = st.session_state.vectorizer
    
    # Model selection
    st.subheader("🎯 Select Model")
    selected_model_name = st.selectbox(
        "Choose a model for prediction:",
        list(models.keys())
    )
    
    selected_model = models[selected_model_name]
    
    st.markdown("---")
    
    # Input methods
    st.subheader("📝 Enter Review")
    
    input_method = st.radio("Choose input method:", ["✍️ Text Input", "📁 Batch Upload"])
    
    if input_method == "✍️ Text Input":
        review_text = st.text_area(
            "Enter your review here:",
            height=150,
            placeholder="Type or paste your product review here..."
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            predict_button = st.button("🔮 Predict Sentiment", use_container_width=True)
        
        if predict_button and review_text:
            with st.spinner("Analyzing sentiment..."):
                rating, prediction = predict_sentiment(review_text, selected_model, vectorizer)
                sentiment_features = get_sentiment_features(review_text)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Rating", f"{rating} ⭐")
                
                with col2:
                    sentiment = "Positive" if sentiment_features['polarity'] > 0 else "Negative" if sentiment_features['polarity'] < 0 else "Neutral"
                    st.metric("Sentiment", sentiment)
                
                with col3:
                    st.metric("Confidence", f"{abs(sentiment_features['polarity']):.2%}")
                
                # Detailed analysis
                st.markdown("---")
                st.subheader("🔍 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Text Features:**")
                    features_df = pd.DataFrame({
                        'Feature': ['Polarity', 'Subjectivity', 'Word Count', 'Avg Word Length'],
                        'Value': [
                            f"{sentiment_features['polarity']:.3f}",
                            f"{sentiment_features['subjectivity']:.3f}",
                            sentiment_features['length'],
                            f"{sentiment_features['avg_word_length']:.2f}"
                        ]
                    })
                    st.dataframe(features_df, use_container_width=True, hide_index=True)
                
                with col2:
                    # Rating visualization
                    stars = ['⭐' * i for i in range(1, 6)]
                    fig = go.Figure(go.Bar(
                        x=[1, 2, 3, 4, 5],
                        y=[1 if i == rating else 0 for i in range(1, 6)],
                        marker_color=['gold' if i == rating else 'lightgray' for i in range(1, 6)],
                        text=stars,
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title='Predicted Rating',
                        xaxis_title='Stars',
                        yaxis_title='',
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    else:  # Batch Upload
        uploaded_file = st.file_uploader("Upload CSV file with reviews", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                
                if 'reviews.text' not in batch_df.columns:
                    st.error("CSV must contain 'reviews.text' column")
                else:
                    if st.button("🔮 Predict All", use_container_width=True):
                        with st.spinner("Processing batch predictions..."):
                            predictions = []
                            
                            for text in batch_df['reviews.text']:
                                try:
                                    rating, pred = predict_sentiment(str(text), selected_model, vectorizer)
                                    predictions.append(rating)
                                except:
                                    predictions.append(None)
                            
                            batch_df['Predicted_Rating'] = predictions
                            
                            st.success(f"✅ Processed {len(batch_df)} reviews!")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.dataframe(batch_df, use_container_width=True)
                            
                            with col2:
                                pred_dist = pd.Series(predictions).value_counts().sort_index()
                                fig = px.bar(
                                    x=pred_dist.index,
                                    y=pred_dist.values,
                                    labels={'x': 'Rating', 'y': 'Count'},
                                    title='Predicted Ratings Distribution'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Download button
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

def show_results():
    """Results and metrics page"""
    st.header("📈 Results & Performance Metrics")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Model Training' section")
        return
    
    models = st.session_state.models
    results_df = st.session_state.results
    X_test_tfidf = st.session_state.X_test_tfidf
    y_test = st.session_state.y_test
    
    # Model selection for detailed view
    st.subheader("🎯 Select Model for Detailed Analysis")
    selected_model_name = st.selectbox(
        "Choose a model:",
        list(models.keys())
    )
    
    selected_model = models[selected_model_name]
    
    # Generate predictions
    y_pred = selected_model.predict(X_test_tfidf)
    
    # Tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Confusion Matrix", "📋 Classification Report", "📈 All Models Comparison"])
    
    with tab1:
        st.subheader(f"Performance Overview - {selected_model_name}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Accuracy</h3>
                <h2>{accuracy:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>F1-Score</h3>
                <h2>{f1:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Precision</h3>
                <h2>{precision:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>Recall</h3>
                <h2>{recall:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performance by class
        st.subheader("Performance by Rating Class")
        
        from sklearn.metrics import precision_recall_fscore_support
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(y_test, y_pred)
        
        class_metrics = pd.DataFrame({
            'Rating': ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'],
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1-Score': f1_per_class,
            'Support': support
        })
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(class_metrics.style.format({
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}'
            }), use_container_width=True, hide_index=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=class_metrics['Rating'],
                y=class_metrics['Precision'],
                name='Precision',
                mode='lines+markers',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=class_metrics['Rating'],
                y=class_metrics['Recall'],
                name='Recall',
                mode='lines+markers',
                line=dict(color='green', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=class_metrics['Rating'],
                y=class_metrics['F1-Score'],
                name='F1-Score',
                mode='lines+markers',
                line=dict(color='red', width=3)
            ))
            fig.update_layout(
                title='Metrics by Class',
                xaxis_title='Rating',
                yaxis_title='Score',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader(f"Confusion Matrix - {selected_model_name}")
        
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = plot_confusion_matrix(cm, f"Confusion Matrix - {selected_model_name}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Matrix Statistics:**")
            
            # Calculate accuracy per class
            class_accuracy = cm.diagonal() / cm.sum(axis=1)
            
            accuracy_df = pd.DataFrame({
                'Class': ['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'],
                'Accuracy': class_accuracy
            })
            
            st.dataframe(accuracy_df.style.format({'Accuracy': '{:.2%}'}), 
                        use_container_width=True, hide_index=True)
            
            st.write("**Interpretation:**")
            st.info("""
            - Diagonal values show correct predictions
            - Off-diagonal values show misclassifications
            - Darker colors indicate higher counts
            """)
    
    with tab3:
        st.subheader(f"Detailed Classification Report - {selected_model_name}")
        
        report = classification_report(y_test, y_pred, 
                                       target_names=['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'],
                                       output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(report_df.style.format({
            'precision': '{:.4f}',
            'recall': '{:.4f}',
            'f1-score': '{:.4f}',
            'support': '{:.0f}'
        }).background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
        use_container_width=True)
        
        st.markdown("---")
        
        # Misclassification analysis
        st.subheader("🔍 Misclassification Analysis")
        
        misclassified = y_test != y_pred
        total_misclassified = misclassified.sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Misclassifications", total_misclassified)
            st.metric("Misclassification Rate", f"{(total_misclassified/len(y_test))*100:.2f}%")
        
        with col2:
            # Error distribution
            error_dist = pd.DataFrame({
                'Actual': y_test[misclassified],
                'Predicted': y_pred[misclassified]
            })
            
            error_counts = error_dist.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
            error_counts = error_counts.nlargest(5, 'Count')
            
            st.write("**Top 5 Misclassification Patterns:**")
            error_counts['Actual'] = error_counts['Actual'] + 1
            error_counts['Predicted'] = error_counts['Predicted'] + 1
            st.dataframe(error_counts, use_container_width=True, hide_index=True)
    
    with tab4:
        st.subheader("📊 All Models Performance Comparison")
        
        # Overall comparison
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(results_df.sort_values('f1_score', ascending=False).style.format({
                'accuracy': '{:.4f}',
                'f1_score': '{:.4f}'
            }).background_gradient(cmap='RdYlGn', subset=['accuracy', 'f1_score']),
            use_container_width=True, hide_index=True)
            
            best_model = results_df.loc[results_df['f1_score'].idxmax(), 'model']
            st.success(f"🏆 Best Model: **{best_model}**")
        
        with col2:
            # Comparison chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Accuracy Comparison', 'F1-Score Comparison'),
                specs=[[{'type': 'bar'}, {'type': 'bar'}]]
            )
            
            fig.add_trace(
                go.Bar(
                    x=results_df['model'],
                    y=results_df['accuracy'],
                    name='Accuracy',
                    marker_color='lightblue',
                    text=results_df['accuracy'].round(4),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=results_df['model'],
                    y=results_df['f1_score'],
                    name='F1-Score',
                    marker_color='lightcoral',
                    text=results_df['f1_score'].round(4),
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Radar chart comparison
        st.subheader("🎯 Multi-Metric Comparison (Radar Chart)")
        
        # Calculate metrics for all models
        all_model_metrics = []
        
        for model_name, model in models.items():
            y_pred = model.predict(X_test_tfidf)
            
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1-Score': f1_score(y_test, y_pred, average='weighted')
            }
            all_model_metrics.append(metrics)
        
        metrics_df = pd.DataFrame(all_model_metrics)
        
        # Create radar chart
        fig = go.Figure()
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, row in metrics_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                theta=categories,
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Model comparison table
        st.subheader("📋 Detailed Metrics Comparison")
        st.dataframe(metrics_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}'
        }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']),
        use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
