import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_sentiment_analysis(symbol):
    """Plot sentiment analysis results"""
    df_sentiment = pd.read_csv(f"{symbol}_sentiment_analysis.csv")
    df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])

    # Sentiment Score Over Time
    plt.figure(figsize=(12, 5))
    sns.lineplot(x=df_sentiment['date'], y=df_sentiment['sentiment_score'], label="Sentiment Score", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.title(f"{symbol} - Sentiment Score Over Time")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.show()

    # Sentiment Class Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_sentiment['sentiment_class'], discrete=True, shrink=0.8, palette="pastel")
    plt.xlabel("Sentiment Class")
    plt.ylabel("Frequency")
    plt.title(f"{symbol} - Sentiment Class Distribution")
    plt.grid(axis="y")
    plt.show()

def plot_predictions(symbol):
    """Plot prediction results"""
    df_predictions = pd.read_csv(f"{symbol}_predictions.csv")
    df_predictions['date'] = pd.to_datetime(df_predictions['date'])

    # Actual vs Predicted Trends
    plt.figure(figsize=(12, 5))
    plt.plot(df_predictions['date'], df_predictions['actual'], label="Actual", marker="o", linestyle="-", color="green")
    plt.plot(df_predictions['date'], df_predictions['predicted'], label="Predicted", marker="s", linestyle="--", color="red")
    plt.xlabel("Date")
    plt.ylabel("Trend Class (0 = Down, 1 = Stable, 2 = Up)")
    plt.title(f"{symbol} - Actual vs Predicted Trends")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(df_predictions['actual'], df_predictions['predicted'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Stable", "Up"])
    
    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{symbol} - Prediction Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    symbol = "BTC-USD"  # Change to your symbol
    plot_sentiment_analysis(symbol)
    plot_predictions(symbol)
