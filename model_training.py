"""
Sentiment Analysis Model Training Script
NLP Project - NUTECH CS22
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization and remove stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def load_and_prepare_data(self, filepath):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        df = pd.read_csv("IMDB Dataset.csv")
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
        
        # Preprocess reviews
        print("\nPreprocessing text data...")
        df['cleaned_review'] = df['review'].apply(self.preprocess_text)
        
        # Convert sentiment to binary (0: negative, 1: positive)
        df['sentiment_binary'] = df['sentiment'].map({'negative': 0, 'positive': 1})
        
        return df
    
    def train_model(self, X_train, y_train, model_type='logistic'):
        """Train the sentiment classification model"""
        print("\nVectorizing text data...")
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        print(f"Training {model_type} model...")
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_vec, y_train)
        print("Model training completed!")
        
        return X_train_vec
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Negative', 'Positive']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def predict(self, text):
        """Predict sentiment for new text"""
        cleaned_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_vec)[0]
        probability = self.model.predict_proba(text_vec)[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probability) * 100
        
        return sentiment, confidence
    
    def save_model(self, model_path='models/sentiment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """Save trained model and vectorizer"""
        import os
        os.makedirs('models', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\nModel saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='models/sentiment_model.pkl', 
                   vectorizer_path='models/vectorizer.pkl'):
        """Load trained model and vectorizer"""
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print("Model and vectorizer loaded successfully!")


def main():
    # Initialize the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Load and prepare data
    df = analyzer.load_and_prepare_data('data/IMDB Dataset.csv')
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], 
        df['sentiment_binary'],
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment_binary']
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    analyzer.train_model(X_train, y_train, model_type='logistic')
    
    # Evaluate model
    metrics = analyzer.evaluate_model(X_test, y_test)
    
    # Save model
    analyzer.save_model()
    
    # Test with sample predictions
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    test_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film. Waste of time and money. Very disappointed.",
        "The acting was decent but the plot was confusing."
    ]
    
    for review in test_reviews:
        sentiment, confidence = analyzer.predict(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f}%)")


if __name__ == "__main__":
    main()