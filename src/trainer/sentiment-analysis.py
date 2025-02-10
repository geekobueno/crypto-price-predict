import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SentimentAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_wiki_data(self, symbol):
        """Load and process Wikipedia edit data"""
        df_wiki = pd.read_csv(f'{symbol}_edits.csv')
        df_wiki['date'] = pd.to_datetime(df_wiki['date'])
        # Aggregate daily edit counts
        df_wiki = df_wiki.groupby('date').agg(
            edit_count=('comment', 'count'),  # Count total edits (rows)
            editor_count=('user', 'nunique')  # Count unique users (editors)
        ).reset_index()
        df_wiki.set_index('date', inplace=True)
        return df_wiki
    
    def load_trends_data(self, symbol):
        """Load and process Google Trends data"""
        df_trends = pd.read_csv(f'{symbol}_google_trends.csv')  # Fixed filename
        df_trends['date'] = pd.to_datetime(df_trends['date'])
        df_trends.set_index('date', inplace=True)

        if 'trend_value' not in df_trends.columns:
            print("Warning: 'trend_value' column is missing in Google Trends data")
            df_trends['trend_value'] = 0  # Default to 0 if missing

        return df_trends
    
    def calculate_sentiment_features(self, symbol):
        """Calculate sentiment features from Wikipedia and Google Trends data"""
        # Load data
        df_wiki = self.load_wiki_data(symbol)
        df_trends = self.load_trends_data(symbol)

        # Handle missing Google Trends data
        if df_trends is None:
            df_trends = pd.DataFrame(index=df_wiki.index)

        # Combine data
        df_sentiment = df_wiki.join(df_trends, how='outer')

        # Forward fill missing values
        df_sentiment = df_sentiment.fillna(method='ffill')

        # Calculate momentum features
        df_sentiment['wiki_momentum'] = df_sentiment['edit_count'].pct_change()
        df_sentiment['wiki_editor_momentum'] = df_sentiment['editor_count'].pct_change()
        df_sentiment['trend_momentum'] = df_sentiment['trend_value'].pct_change()

        # Calculate rolling averages
        df_sentiment['wiki_7d_avg'] = df_sentiment['edit_count'].rolling(7).mean()
        df_sentiment['trend_7d_avg'] = df_sentiment['trend_value'].rolling(7).mean()

        # Calculate volatility
        df_sentiment['wiki_volatility'] = df_sentiment['edit_count'].rolling(7).std()
        df_sentiment['trend_volatility'] = df_sentiment['trend_value'].rolling(7).std()

        # Scale features
        features_to_scale = ['edit_count', 'editor_count', 'trend_value',
                           'wiki_7d_avg', 'trend_7d_avg',
                           'wiki_volatility', 'trend_volatility']

        df_sentiment[features_to_scale] = self.scaler.fit_transform(
            df_sentiment[features_to_scale].fillna(0)
        )

        return df_sentiment.dropna()
    
    def get_sentiment_score(self, row):
        """Calculate a combined sentiment score"""
        # Weighted combination of different signals
        wiki_weight = 0.3
        trend_weight = 0.4
        momentum_weight = 0.3
        
        wiki_signal = (row['edit_count'] + row['wiki_7d_avg']) / 2
        trend_signal = (row['trend_value'] + row['trend_7d_avg']) / 2
        momentum_signal = (row['wiki_momentum'] + row['trend_momentum']) / 2
        
        score = (wiki_weight * wiki_signal +
                trend_weight * trend_signal +
                momentum_weight * momentum_signal)
        
        return score
    
    def analyze_sentiment(self, symbol):
        """Perform complete sentiment analysis"""
        df_sentiment = self.calculate_sentiment_features(symbol)

        # Add combined sentiment score
        df_sentiment['sentiment_score'] = df_sentiment.apply(
            self.get_sentiment_score, axis=1
        )

        # Classify sentiment
        df_sentiment['sentiment_class'] = pd.qcut(
            df_sentiment['sentiment_score'],
            q=3,
            labels=['negative', 'neutral', 'positive']
        )

        # Save results to CSV
        df_sentiment.to_csv(f'{symbol}_sentiment_analysis.csv')
        print(f"Sentiment analysis saved to {symbol}_sentiment_analysis.csv")

        return df_sentiment

if __name__ == '__main__':
    # Example usage
    analyzer = SentimentAnalyzer()
    symbol = 'BTC-USD'
    sentiment_data = analyzer.analyze_sentiment(symbol)
    print(sentiment_data.head())
