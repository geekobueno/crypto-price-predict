# Crypto Price Prediction with AI & Twitter Sentiment Analysis

## 📌 Project Overview

This project aims to predict cryptocurrency price movements by combining **historical price data** with **Twitter sentiment analysis**. The goal is to determine optimal buy/sell signals by analyzing past trends and real-time market sentiment from social media.

## 🔥 Key Features

✅ **Historical Data Analysis:** Using Kaggle’s [Top 1000 Cryptos Historical](https://www.kaggle.com/datasets/ayushkhaire/top-1000-cryptos-historical/data) dataset.  
✅ **Twitter Sentiment Scraping:** Extracting trending crypto-related hashtags to gauge market sentiment.  
✅ **Machine Learning Models:** Testing multiple ML models (Random Forest, XGBoost, LSTM, ARIMA, Prophet) for price forecasting.  
✅ **AI-Powered Insights:** Integrating GPT API for generating trading insights based on data trends.  
✅ **Buy/Sell Predictions:** A classification model to determine whether to `BUY (1)`, `HOLD (0)`, or `SELL (-1)`.

---

## 📊 Data Collection & Processing

### **1. Data Sources**

- **Historical Price Data:** Kaggle dataset (OHLCV format: Open, High, Low, Close, Volume).
- **Twitter Data:** Scraped using `snscrape` or `Tweepy` (hashtags: `#Bitcoin`, `#Ethereum`, etc.).

### **2. Preprocessing Steps**

- Cleaning missing values and outliers.
- Feature Engineering:
  - Moving Averages (7-day, 30-day)
  - Volatility Indicators
  - Sentiment Scores (derived from Twitter data using TextBlob or VADER)

---

## 🧠 Model Selection & Training

### **1. Time-Series & Regression Models**

- **Random Forest**
- **XGBoost**
- **ARIMA** / **Prophet** (for long-term trends)
- **LSTM** (for deep learning-based sequential forecasting)

### **2. Classification Model for Trading Signals**

- Predicting `BUY`, `HOLD`, or `SELL` decisions.
- Training using sentiment + historical indicators.

### **3. AI Integration (GPT API)**

- GPT-4 API will:
  - Analyze historical + sentiment trends.
  - Provide market insights & possible trading strategies.
  - Summarize trends based on model outputs.

---



## ⚡ Technologies Used

- **Python** (Pandas, NumPy, Scikit-Learn, TensorFlow, Statsmodels, Prophet)
- **Twitter Scraping** (`snscrape`, `Tweepy`)
- **Natural Language Processing** (VADER, TextBlob, GPT API)
- **Machine Learning** (Random Forest, XGBoost, LSTM)
- **Visualization** (Matplotlib, Seaborn, Plotly)

---

## 📌 Expected Outcomes

✔ A working ML model predicting crypto price movements.
✔ AI-powered buy/sell recommendations.
✔ A final report analyzing performance & insights.

---

## 📢 Future Improvements

🔹 Integrate **real-time Twitter sentiment streaming** for live analysis.  
🔹 Improve model accuracy by fine-tuning hyperparameters.  
🔹 Add **news sentiment analysis** from financial sites (e.g., CoinDesk, Bloomberg).

---

## 📜 License

MIT License (To Be Decided)

---

## 🤝 Contributing

This is a provisional project. Contributions & feedback are welcome!

**Author:** Martin Simtaya  
**Date:** January 30, 2025
