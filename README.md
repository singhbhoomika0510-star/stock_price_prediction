# 📈 StockPulse ML - Stock Price Prediction Mini Project

A modern, responsive, and beginner-friendly **Stock Price Prediction Machine Learning Web Application** built with a **warm, light aesthetic** (cream, beige, soft brown, and subtle green accents).

Designed with complete mathematical transparency, **Linear Regression** as the primary explainable model, **zero lookahead bias / data leakage**, and a comprehensive **college viva preparation guide**.

---

## 🌟 Key Features

* **Warm, Light Human-Designed Theme**: Styled with a soothing cream/beige backdrop (`#FAF7F2`), rounded cards, clear typography, and subtle sage green accents. Free from aggressive neon gradients and dark AI slop.
* **Live Market Data**: Directly fetches adjusted daily Open, High, Low, Close, and Volume data via `yfinance` (supports US tickers like `AAPL`, `MSFT`, `TSLA`, `NVDA`, and Indian NSE tickers like `RELIANCE.NS`, `TCS.NS`, `INFY.NS`).
* **Zero Data Leakage**: Enforces a strict **80/20 chronological split** (`shuffle=False`). Unlike random shuffling, past data only ever trains future predictions.
* **Explainable Linear Regression**: Provides full feature weights (coefficients $\beta_i$) and intercept $\beta_0$, demonstrating how each price lag and moving average contributes to tomorrow's price.
* **Multi-Horizon Forecasting**: Allows forecasting for **1 day (Tomorrow)**, **7 days (1 week)**, **14 days (2 weeks)**, or **30 days (1 month)** ahead via autoregressive rollout.
* **Comprehensive Metrics**:
  * **MAE** (Mean Absolute Error)
  * **RMSE** (Root Mean Squared Error)
  * **MSE** (Mean Squared Error)
  * **$R^2$ Score** (Coefficient of Determination)
  * **Directional Accuracy** (% of days the model anticipated Up/Down movement correctly)
* **Multi-View Interactive Charts (Chart.js)**:
  * Full Historical + Forecast view
  * Test Fit view (Unseen actual vs predicted overlay)
  * Forecast trajectory view
* **College Viva & Oral Exam Guide**: Built-in answers to typical examiner questions (data leakage, OLS formula, LSTM vs Regression, Efficient Market Hypothesis).
* **Graceful Offline Fallback**: If Yahoo Finance is unreachable or rate-limited, the system seamlessly uses a realistic random-walk generator so your college presentation never crashes.

---

## 🏗️ Project Architecture

```
stock price/
├── app.py                   # Flask server providing Web UI and REST API
├── ml_engine.py             # Machine learning pipeline, feature engineering, and forecasting
├── requirements.txt         # Python dependencies
├── models/                  # Saved model files (.joblib)
├── templates/
│   └── index.html           # Main responsive HTML5 single-page application
└── static/
    ├── css/
    │   └── style.css        # Warm light palette styling & layout
    └── js/
        └── main.js          # Chart.js integration, API calls, and UI updates
```

---

## 🚀 How to Run the Application

### 1. Ensure Dependencies are Installed
Open your terminal in the project directory:
```bash
pip install -r requirements.txt
```

### 2. Start the Flask Server
```bash
python app.py
```

### 3. Open in Browser
Navigate to:
```
http://127.0.0.1:5000
```
The application will automatically load a sample demonstration (`AAPL`) so you can explore the features right away!

---

## 🧪 Machine Learning Pipeline Explained

### 1. Feature Engineering
We construct autoregressive features using strictly past information:
* `Lag_1`: Yesterday's Close price ($P_{t-1}$)
* `Lag_2`: Close price 2 days ago ($P_{t-2}$)
* `Lag_3`: Close price 3 days ago ($P_{t-3}$)
* `SMA_5`: 5-day Simple Moving Average
* `SMA_10`: 10-day Simple Moving Average
* `SMA_20`: 20-day Simple Moving Average
* `Volatility_Pct`: $\frac{\text{High} - \text{Low}}{\text{Close}} \times 100$
* `Volume_Change_Pct`: Day-over-day trading volume change percentage
* **Target ($y$):** Next day's Close price ($P_{t+1}$)

### 2. Model Training
We use **Ordinary Least Squares (OLS) Linear Regression**:
$$\hat{y} = \beta_0 + \sum_{i=1}^{k} \beta_i X_i$$

### 3. Model Persistence
Trained models are automatically serialized and saved to `models/{symbol}_linear_model.joblib` using `joblib`.

---

## 🎓 Viva Questions & Answers Cheat Sheet

1. **Why Linear Regression?**
   > *Answer:* It provides complete transparency. Each feature has an explicit coefficient, making it easy to explain to stakeholders. It trains in milliseconds and avoids overfitting common in deep neural networks.

2. **How did you prevent Data Leakage?**
   > *Answer:* In time-series analysis, random shuffling leaks future prices into past predictions. We enforced a strict chronological split (`shuffle=False`): the first 80% of historical dates train the model, and the final 20% serve as the unseen test set.

3. **What is the difference between MAE and RMSE?**
   > *Answer:* MAE is the direct average dollar error. RMSE squares errors before averaging, making it penalize large forecast misses much more severely.

4. **Can this model guarantee profit in real trading?**
   > *Answer:* No. Stock prices are influenced by unforeseen corporate earnings, breaking news, interest rates, and human emotion. Machine learning offers probabilistic estimations, not guarantees.

---

## 📄 License & Disclaimer
This project is created for **academic, educational, and mini-project purposes only**. Predictions are statistical approximations and do not constitute investment advice.
