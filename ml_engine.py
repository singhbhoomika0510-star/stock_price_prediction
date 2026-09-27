"""
ml_engine.py
============
Core Machine Learning Engine for Stock Price Prediction using Linear Regression.

Key Capabilities:
1. Historical data acquisition via Yahoo Finance (yfinance) with fallback generator.
2. Temporal feature engineering (Lag prices, Moving Averages, Volatility).
3. Chronological train-test split (eliminating data leakage).
4. Linear Regression training, evaluation (MAE, MSE, RMSE, R2, Directional Accuracy).
5. Multi-step autoregressive future forecasting for 1 to 30 days.
6. Model persistence using joblib.
"""

import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Directory to persist trained models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Curated popular stocks with readable labels and currency signs (Rupees as primary currency)
POPULAR_STOCKS = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "exchange": "NSE", "currency": "₹"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services (TCS)", "exchange": "NSE", "currency": "₹"},
    {"symbol": "INFY.NS", "name": "Infosys", "exchange": "NSE", "currency": "₹"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "exchange": "NSE", "currency": "₹"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "exchange": "NSE", "currency": "₹"},
    {"symbol": "SBIN.NS", "name": "State Bank of India (SBI)", "exchange": "NSE", "currency": "₹"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "exchange": "NSE", "currency": "₹"},
    {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "currency": "₹"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "currency": "₹"}
]

FEATURE_COLUMNS = ['Lag_1', 'Lag_2', 'Lag_3', 'SMA_5', 'SMA_10', 'SMA_20', 'Volatility_Pct', 'Volume_Change_Pct']


def get_stock_currency(symbol: str) -> str:
    """Always use Indian Rupee (₹) as the primary project currency."""
    return "₹"


def generate_fallback_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Generates realistic synthetic stock data if Yahoo Finance is unreachable or rate-limited.
    Ensures seamless demonstration during viva or offline presentations.
    """
    days_map = {"6mo": 126, "1y": 252, "2y": 504, "5y": 1260}
    num_days = days_map.get(period, 252)

    base_prices = {
        "AAPL": 230.0, "MSFT": 420.0, "GOOGL": 175.0, "AMZN": 185.0,
        "TSLA": 250.0, "NVDA": 125.0, "RELIANCE.NS": 2950.0, "TCS.NS": 4200.0,
        "INFY.NS": 1850.0, "HDFCBANK.NS": 1650.0
    }
    start_price = base_prices.get(symbol.upper(), 150.0)

    # Generate dates (weekdays only)
    end_date = datetime.date.today()
    dates = pd.bdate_range(end=end_date, periods=num_days)

    # Random walk with slight positive drift (GBM-like)
    np.random.seed(abs(hash(symbol)) % 10000)
    returns = np.random.normal(loc=0.0004, scale=0.015, size=num_days)
    price_series = start_price * np.exp(np.cumsum(returns))

    high_series = price_series * (1 + np.random.uniform(0.005, 0.02, size=num_days))
    low_series = price_series * (1 - np.random.uniform(0.005, 0.02, size=num_days))
    open_series = price_series * (1 + np.random.uniform(-0.01, 0.01, size=num_days))
    volume_series = np.random.randint(10_000_000, 60_000_000, size=num_days)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_series,
        "High": high_series,
        "Low": low_series,
        "Close": price_series,
        "Volume": volume_series
    })
    return df


def fetch_stock_data(symbol: str, period: str = "1y") -> tuple[pd.DataFrame, dict]:
    """
    Fetches historical stock prices using yfinance.
    Cleans multi-index headers and handles missing dates.
    Falls back gracefully to realistic simulated series if network/rate-limit fails.
    """
    clean_symbol = symbol.strip().upper()
    metadata = {
        "symbol": clean_symbol,
        "period": period,
        "currency": get_stock_currency(clean_symbol),
        "is_simulated": False,
        "company_name": clean_symbol
    }

    # Match predefined names if available
    for stock in POPULAR_STOCKS:
        if stock["symbol"].upper() == clean_symbol:
            metadata["company_name"] = stock["name"]
            metadata["currency"] = stock["currency"]
            break

    try:
        # Download historical data
        raw_df = yf.download(clean_symbol, period=period, interval="1d", progress=False, auto_adjust=True)
        
        if raw_df is None or raw_df.empty or len(raw_df) < 30:
            raise ValueError(f"Insufficient historical data returned for {clean_symbol}.")

        # Handle yfinance multi-index column headers
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)

        df = raw_df.reset_index()

        # Standardize column naming
        df.rename(columns={col: col.capitalize() for col in df.columns}, inplace=True)
        
        if 'Date' not in df.columns and 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)

        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing essential column '{col}' in downloaded data.")

        df = df[required_cols].copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.dropna().reset_index(drop=True)

        # Try fetching ticker company name if not found
        if metadata["company_name"] == clean_symbol:
            try:
                ticker_obj = yf.Ticker(clean_symbol)
                info = ticker_obj.info
                if info and "shortName" in info:
                    metadata["company_name"] = info["shortName"]
                elif info and "longName" in info:
                    metadata["company_name"] = info["longName"]
            except Exception:
                pass

        return df, metadata

    except Exception as e:
        print(f"[Warning] Failed fetching live data for {clean_symbol} ({e}). Utilizing fallback dataset.")
        df = generate_fallback_data(clean_symbol, period=period)
        metadata["is_simulated"] = True
        return df, metadata


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs predictive features strictly using past observations to prevent data leakage.
    Target: Next Day Close Price (y_t = Close_{t+1})
    Features:
      - Lag_1: Yesterday's Close
      - Lag_2: Day before yesterday's Close
      - Lag_3: 3 days ago Close
      - SMA_5: 5-Day Simple Moving Average
      - SMA_10: 10-Day Simple Moving Average
      - SMA_20: 20-Day Simple Moving Average
      - Volatility_Pct: (High - Low) / Close * 100
      - Volume_Change_Pct: Day-over-day Volume percentage change
    """
    data = df.copy()

    # Sort chronologically
    data = data.sort_values('Date').reset_index(drop=True)

    # 1. Price Lags
    data['Lag_1'] = data['Close']
    data['Lag_2'] = data['Close'].shift(1)
    data['Lag_3'] = data['Close'].shift(2)

    # 2. Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # 3. Volatility and Volume Momentum (safe against division by zero)
    safe_close = data['Close'].replace(0, np.nan)
    data['Volatility_Pct'] = (((data['High'] - data['Low']) / safe_close) * 100).replace([np.inf, -np.inf], 0).fillna(0)
    data['Volatility_Pct'] = data['Volatility_Pct'].clip(0, 100)

    safe_volume = data['Volume'].replace(0, np.nan).ffill().bfill().fillna(1)
    data['Volume_Change_Pct'] = (safe_volume.pct_change().replace([np.inf, -np.inf], 0).fillna(0) * 100).clip(-100, 500)

    # 4. Target Variable: Next Day's Close Price (shifted backward by 1)
    data['Target_Next_Close'] = data['Close'].shift(-1)

    # Replace any potential infs or -infs with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop incomplete rows (first 20 rows due to 20-day SMA, and last row due to target shift)
    data_clean = data.dropna().reset_index(drop=True)
    return data_clean


def train_linear_model(data_featured: pd.DataFrame, symbol: str) -> dict:
    """
    Trains a Scikit-Learn Linear Regression model using a chronological train/test split.
    Prevents lookahead bias / data leakage.
    Evaluates model performance: MAE, MSE, RMSE, R2, Directional Accuracy.
    Saves trained model to disk.
    """
    X = data_featured[FEATURE_COLUMNS]
    y = data_featured['Target_Next_Close']
    dates = data_featured['Date']

    # Chronological Split (80% Train, 20% Test) - Shuffle is strictly False!
    split_index = int(len(data_featured) * 0.80)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dates_train, dates_test = dates.iloc[:split_index], dates.iloc[split_index:]

    # Model Initialization & Fitting
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on Unseen Test Dataset
    y_pred_test = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred_test))

    # Directional Accuracy: Did model predict UP/DOWN correctly relative to today's close?
    actual_direction = np.sign(y_test.values - X_test['Lag_1'].values)
    pred_direction = np.sign(y_pred_test - X_test['Lag_1'].values)
    directional_accuracy = float(np.mean(actual_direction == pred_direction) * 100)

    # Feature Importance (Coefficients)
    coefficients = [
        {"feature": feat, "weight": round(float(coef), 4), "impact": "Positive" if coef > 0 else "Negative"}
        for feat, coef in zip(FEATURE_COLUMNS, model.coef_)
    ]
    # Sort coefficients by absolute magnitude
    coefficients = sorted(coefficients, key=lambda x: abs(x["weight"]), reverse=True)

    # Save model file
    safe_symbol = "".join([c if c.isalnum() else "_" for c in symbol])
    model_filepath = os.path.join(MODELS_DIR, f"{safe_symbol}_linear_model.joblib")
    joblib.dump(model, model_filepath)

    return {
        "model": model,
        "model_file": model_filepath,
        "intercept": round(float(model.intercept_), 4),
        "coefficients": coefficients,
        "metrics": {
            "mae": round(mae, 2),
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "directional_accuracy": round(directional_accuracy, 1),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "test_results": {
            "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in dates_test],
            "actual": [round(float(v), 2) for v in y_test.values],
            "predicted": [round(float(v), 2) for v in y_pred_test]
        }
    }


def forecast_future(model: LinearRegression, df_raw: pd.DataFrame, days_ahead: int = 7) -> dict:
    """
    Performs recursive multi-step forecasting for the upcoming N trading days.
    Uses the latest available data window and projects forward.
    """
    days_ahead = min(max(int(days_ahead), 1), 30)
    
    # Recent history of close prices
    close_prices = df_raw['Close'].tolist()
    last_date = df_raw['Date'].iloc[-1]
    if isinstance(last_date, str):
        last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d").date()
    elif isinstance(last_date, pd.Timestamp):
        last_date = last_date.date()

    # Generate business days for future predictions
    future_dates = pd.bdate_range(start=last_date + datetime.timedelta(days=1), periods=days_ahead)

    future_predictions = []
    working_closes = list(close_prices)

    # Estimate average volatility and volume change from recent 10 days
    recent_volatility = float(((df_raw['High'] - df_raw['Low']) / df_raw['Close'] * 100).tail(10).mean())
    recent_vol_change = 0.0

    for i in range(days_ahead):
        # Build features for current step
        lag_1 = working_closes[-1]
        lag_2 = working_closes[-2] if len(working_closes) >= 2 else lag_1
        lag_3 = working_closes[-3] if len(working_closes) >= 3 else lag_2

        sma_5 = np.mean(working_closes[-5:])
        sma_10 = np.mean(working_closes[-10:])
        sma_20 = np.mean(working_closes[-20:])

        input_features = pd.DataFrame([{
            'Lag_1': lag_1,
            'Lag_2': lag_2,
            'Lag_3': lag_3,
            'SMA_5': sma_5,
            'SMA_10': sma_10,
            'SMA_20': sma_20,
            'Volatility_Pct': recent_volatility,
            'Volume_Change_Pct': recent_vol_change
        }])

        pred_price = float(model.predict(input_features)[0])
        
        # Soft safeguard against negative prices
        pred_price = max(pred_price, 0.01)

        future_predictions.append(round(pred_price, 2))
        working_closes.append(pred_price)

    current_price = round(float(close_prices[-1]), 2)
    final_predicted_price = future_predictions[-1]
    price_diff = round(final_predicted_price - current_price, 2)
    pct_change = round((price_diff / current_price) * 100, 2)

    if pct_change > 0.5:
        sentiment = "Bullish"
        trend_class = "positive"
        trend_icon = "bi-arrow-up-right"
    elif pct_change < -0.5:
        sentiment = "Bearish"
        trend_class = "negative"
        trend_icon = "bi-arrow-down-right"
    else:
        sentiment = "Neutral / Stable"
        trend_class = "neutral"
        trend_icon = "bi-arrow-right"

    return {
        "horizon_days": days_ahead,
        "target_date": future_dates[-1].strftime("%Y-%m-%d"),
        "current_price": current_price,
        "predicted_price": final_predicted_price,
        "price_diff": price_diff,
        "pct_change": pct_change,
        "sentiment": sentiment,
        "trend_class": trend_class,
        "trend_icon": trend_icon,
        "trajectory": [
            {"date": d.strftime("%Y-%m-%d"), "price": p}
            for d, p in zip(future_dates, future_predictions)
        ]
    }


def run_full_pipeline(symbol: str = "RELIANCE.NS", period: str = "1y", days_ahead: int = 7) -> dict:
    """
    End-to-end wrapper combining data ingestion, feature extraction,
    training, evaluation, and future projection.
    """
    # 1. Fetch
    df_raw, metadata = fetch_stock_data(symbol, period=period)

    # 2. Features
    df_featured = engineer_features(df_raw)

    if len(df_featured) < 25:
        raise ValueError("The dataset has too few records for model training after lag engineering.")

    # 3. Train & Evaluate
    train_results = train_linear_model(df_featured, symbol)

    # 4. Future Forecast
    forecast_results = forecast_future(train_results["model"], df_raw, days_ahead=days_ahead)

    # 5. Prepare historical chart slice (e.g. past 120 points for chart clarity)
    hist_slice = df_raw.tail(150)
    historical_chart = {
        "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in hist_slice['Date']],
        "prices": [round(float(p), 2) for p in hist_slice['Close']]
    }

    return {
        "status": "success",
        "metadata": metadata,
        "metrics": train_results["metrics"],
        "intercept": train_results["intercept"],
        "coefficients": train_results["coefficients"],
        "test_results": train_results["test_results"],
        "forecast": forecast_results,
        "historical_chart": historical_chart,
        "model_file": train_results["model_file"]
    }
