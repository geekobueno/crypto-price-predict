import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
import os

class CryptoPricePredictor:
    def __init__(self, model_name="distilbert-base-uncased", batch_size=8):
        """
        Initialize with a lightweight model suitable for CPU
        distilbert-base-uncased is about 260MB and runs well on CPU
        """
        self.batch_size = batch_size
        self.model_name = model_name
        self.max_length = 128  # Reduced sequence length for memory efficiency
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,  # Price up, down, or stable
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
        
        # Initialize scaler
        self.scaler = MinMaxScaler()
        
    def load_market_data(self, symbol):
        """Load market data with technical indicators"""
        df_market = pd.read_csv(f'{symbol}_with_indicators.csv')
        df_market['date'] = pd.to_datetime(df_market['date'])  # Fixed column name
        df_market.set_index('date', inplace=True)
        return df_market

    def load_sentiment_data(self, symbol):
        """Load previously saved sentiment analysis results"""
        try:
            df_sentiment = pd.read_csv(f'{symbol}_sentiment_analysis.csv')
            df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
            df_sentiment.set_index('date', inplace=True)
            return df_sentiment
        except FileNotFoundError:
            print(f"Error: Sentiment data file '{symbol}_sentiment_analysis.csv' not found.")
            return None

    def prepare_market_data(self, df):
        """Convert market data into a format suitable for the transformer"""
        # Select and scale features
        feature_cols = [
            'close', 'volume', 'SMA_20', 'RSI', 'MACD',
            'BB_Middle', 'Stoch_K', 'trend_value'
        ]
        
        df = df.dropna(subset=feature_cols)  # Ensure no NaNs
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=feature_cols, index=df.index)
        
        # Create text descriptions
        texts = [
            f"price: {row['close']:.4f} volume: {row['volume']:.4f} "
            f"sma20: {row['SMA_20']:.4f} rsi: {row['RSI']:.4f} "
            f"macd: {row['MACD']:.4f} bb: {row['BB_Middle']:.4f} "
            f"stoch: {row['Stoch_K']:.4f} trend: {row['trend_value']:.4f}"
            for _, row in scaled_df.iterrows()
        ]

        # Create labels (1: up, 0: stable, -1: down)
        returns = df['close'].pct_change().shift(-1)
        labels = np.where(returns > 0.01, 2, np.where(returns < -0.01, 0, 1))

        return texts[:-1], labels[:-1]  # Remove last point to match labels length
    
    def create_dataset(self, texts, labels):
        """Create a HuggingFace dataset"""
        dataset = Dataset.from_dict({'text': texts, 'label': labels})

        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )

        return dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    def train(self, train_dataset, val_dataset=None, num_epochs=3):
        """Train the model and resume from last checkpoint if available"""

        checkpoint_dir = "./crypto_results/checkpoint-100"

        training_args = TrainingArguments(
            output_dir="./crypto_results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            evaluation_strategy="steps" if val_dataset else "no",
            save_steps=100,
            eval_steps=100,
            logging_dir='./logs',
            logging_steps=10,
            weight_decay=0.01,
            warmup_steps=100,
            load_best_model_at_end=True if val_dataset else False,
            gradient_accumulation_steps=4,
            fp16=False,
            dataloader_num_workers=0,
            save_total_limit=2,  # Keep last 2 checkpoints to avoid excessive storage
            resume_from_checkpoint=checkpoint_dir if os.path.exists(checkpoint_dir) else None
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train(resume_from_checkpoint=checkpoint_dir if os.path.exists(checkpoint_dir) else None)
        trainer.save_model("./crypto_model_final")
    
    def predict(self, text_data):
        """Make predictions on new data"""
        inputs = self.tokenizer(
            text_data,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            
        return predictions.cpu().numpy()

def run_analysis(symbol):
    """Run complete analysis pipeline"""
    print(f"Starting analysis for {symbol}...")
    
    # Initialize predictor
    predictor = CryptoPricePredictor(batch_size=4)
    
    # Load market data
    df_market = predictor.load_market_data(symbol)

    # Load sentiment data from CSV
    df_sentiment = predictor.load_sentiment_data(symbol)

    if df_sentiment is None:
        print("Sentiment data is missing. Exiting analysis.")
        return None, None, None

    # Merge market and sentiment data
    df = df_market.join(df_sentiment, how='inner')
    
    # Prepare data for model
    texts, labels = predictor.prepare_market_data(df)
    
    # Split data
    split_idx = int(len(texts) * 0.8)
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets
    train_dataset = predictor.create_dataset(train_texts, train_labels)
    val_dataset = predictor.create_dataset(val_texts, val_labels)
    
    # Train model
    predictor.train(train_dataset, val_dataset, num_epochs=3)
    
    # Make predictions
    predictions = predictor.predict(val_texts)

    # Save results to CSV
   # Ensure lengths match
    min_length = min(len(df.index[split_idx:]), len(val_labels), len(predictions))
    df_results = pd.DataFrame({
        'date': df.index[split_idx:split_idx + min_length],  # Match shortest array length
        'actual': val_labels[:min_length],
        'predicted': np.argmax(predictions, axis=1)[:min_length]
    })
    df_results.to_csv(f"{symbol}_predictions.csv", index=False)
    
    print(f"Predictions saved to {symbol}_predictions.csv")
    
    return predictor, predictions, val_labels

if __name__ == '__main__':
    symbol = 'BTC-USD'
    predictor, predictions, actual = run_analysis(symbol)
