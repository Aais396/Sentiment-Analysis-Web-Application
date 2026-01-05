"""
Sentiment Analysis Web Application
NLP Project - NUTECH CS22
Interactive Interface using Streamlit
"""

import streamlit as st
import pickle
import re
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3rem;
        font-size: 18px;
    }
    .positive {
        color: #28a745;
        font-size: 24px;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

class SentimentPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            with open('models/sentiment_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def predict_sentiment(self, text):
        """Predict sentiment and return result with confidence"""
        if not text.strip():
            return None, None, None
        
        cleaned_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability) * 100
        
        return sentiment, confidence, probability

# Initialize the predictor
@st.cache_resource
def get_predictor():
    predictor = SentimentPredictor()
    if predictor.load_models():
        return predictor
    return None

def create_gauge_chart(confidence, sentiment):
    """Create a gauge chart for confidence visualization"""
    color = "#28a745" if sentiment == "Positive" else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9e6'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_probability_chart(probability):
    """Create a bar chart showing probabilities for both sentiments"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Negative', 'Positive'],
            y=[probability[0]*100, probability[1]*100],
            marker_color=['#dc3545', '#28a745'],
            text=[f'{probability[0]*100:.2f}%', f'{probability[1]*100:.2f}%'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Probability Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def main():
    # Header
    st.title("🎭 Sentiment Analysis System")
    st.markdown("### Analyze the sentiment of text using Machine Learning")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About This Project")
        st.info("""
        **NLP Project - NUTECH**
        
        **Course:** Natural Language Processing
        
        **Batch:** CS22 | **Semester:** 7
        
        **Category:** Text Classification
        
        **Task:** Sentiment Analysis
        
        This system analyzes text and determines whether the sentiment is **Positive** or **Negative**.
        """)
        
        st.header("🎯 Features")
        st.markdown("""
        - Real-time sentiment prediction
        - Confidence score visualization
        - Probability distribution
        - Batch text analysis
        - Model performance metrics
        """)
        
        st.header("👥 Team Members")
        st.text("1. Muhammad Awais\n" 
        "2. Muhammad Zohaib Arif\n" \
        "3. Hamad Ali\n" \
        "4. Amir Ali khaskhali")
        
    # Load the predictor
    predictor = get_predictor()
    
    if predictor is None:
        st.error("❌ Models not found! Please train the model first.")
        st.info("Run `python train_model.py` to train the model.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Analysis", "📋 Batch Analysis", "📈 Model Info"])
    
    with tab1:
        st.header("Analyze Single Text")
        
        # Text input
        user_input = st.text_area(
            "Enter your text here:",
            height=150,
            placeholder="Type or paste your text here... (e.g., movie review, product feedback, etc.)"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
        
        if analyze_button and user_input:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence, probability = predictor.predict_sentiment(user_input)
                
                if sentiment:
                    st.success("✅ Analysis Complete!")
                    
                    # Display result
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if sentiment == "Positive":
                            st.markdown(f'<p class="positive">😊 {sentiment} Sentiment</p>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p class="negative">😞 {sentiment} Sentiment</p>', 
                                      unsafe_allow_html=True)
                        
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                        
                    with col2:
                        st.plotly_chart(create_gauge_chart(confidence, sentiment), 
                                      use_container_width=True)
                    
                    # Probability chart
                    st.plotly_chart(create_probability_chart(probability), 
                                  use_container_width=True)
                    
                    # Additional insights
                    st.markdown("---")
                    st.subheader("💡 Insights")
                    
                    word_count = len(user_input.split())
                    char_count = len(user_input)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Word Count", word_count)
                    col2.metric("Character Count", char_count)
                    col3.metric("Sentiment Strength", 
                              "Strong" if confidence > 80 else "Moderate" if confidence > 60 else "Weak")
    
    with tab2:
        st.header("Batch Analysis")
        st.markdown("Analyze multiple texts at once")
        
        uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", 
                                        type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if 'text' in df.columns:
                if st.button("Analyze All Texts"):
                    with st.spinner("Analyzing..."):
                        results = []
                        for text in df['text']:
                            sentiment, confidence, _ = predictor.predict_sentiment(str(text))
                            results.append({
                                'text': text,
                                'sentiment': sentiment,
                                'confidence': f"{confidence:.2f}%"
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.success("✅ Batch analysis complete!")
                        st.dataframe(results_df)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
            else:
                st.error("❌ CSV must contain a 'text' column!")
    
    with tab3:
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Model Details")
            st.markdown("""
            - **Algorithm:** Logistic Regression
            - **Vectorization:** TF-IDF
            - **Features:** 5000 max features
            - **N-grams:** Unigrams and Bigrams
            - **Dataset:** IMDB Movie Reviews (50,000 reviews)
            """)
        
        with col2:
            st.subheader("🎯 Performance Metrics")
            st.markdown("""
            - **Accuracy:** ~89%
            - **Precision:** ~88%
            - **Recall:** ~90%
            - **F1-Score:** ~89%
            
            
            """)
        
        st.subheader("🔧 Preprocessing Steps")
        st.markdown("""
        1. Convert text to lowercase
        2. Remove HTML tags and URLs
        3. Remove special characters and digits
        4. Tokenization
        5. Remove stopwords
        6. Lemmatization
        7. TF-IDF vectorization
        """)
        
        st.subheader("📚 Dataset Information")
        st.markdown("""
        **Source:** IMDB Movie Reviews Dataset
        - Total Reviews: 50,000
        - Positive Reviews: 25,000
        - Negative Reviews: 25,000
        - Training Set: 40,000 (80%)
        - Test Set: 10,000 (20%)
        """)

if __name__ == "__main__":
    main()