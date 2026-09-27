import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

print("1. Data Collection: Fetching historical data for RELIANCE (RELIANCE.NS)...")
reliance = yf.Ticker("RELIANCE.NS")
df = reliance.history(period="5y")

print("2. Data Preprocessing & Feature Engineering...")
df.dropna(inplace=True)

df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

df['Target_Next_Day'] = df['Close'].shift(-1)
df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Signal_Line']
X = df[features]
y = df['Target_Next_Day']

print("3. Model Development & Training...")
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df.index[split_idx:]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

print("4. Testing & Performance Evaluation...")
y_pred = rf_model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluation Results ---")
print(f"Mean Absolute Error (MAE): Rs.{mae:.2f}")
print(f"Root Mean Squared Error (RMSE): Rs.{rmse:.2f}")
print(f"R² Score: {r2:.2f}")

print("\n5. Result Analysis: Saving Plot and Results...")
plt.figure(figsize=(12,6))
plt.plot(dates_test, y_test.values, label='Actual Price', color='#f43f5e')
plt.plot(dates_test, y_pred, label='Predicted Price', color='#2dd4bf', alpha=0.8)
plt.title('RELIANCE Stock Price: Actual vs Predicted (Random Forest)')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('actual_vs_predicted.png')
print("Saved plot to 'actual_vs_predicted.png'.")

results = {
    "Stock": "RELIANCE.NS",
    "Model": "Random Forest Regressor",
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2_Score": round(r2, 4)
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Saved results to 'results.json'.")
print("ML Pipeline Completed Successfully!")
