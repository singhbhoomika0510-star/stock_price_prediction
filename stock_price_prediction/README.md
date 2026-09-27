# Stock Price Prediction (ML Lab)

This is an end-to-end Machine Learning Application built with Streamlit for Stock Price Prediction.

## Features
- **Dataset Upload:** Upload your own CSV stock dataset.
- **Data Preprocessing:** Handles missing values and sorts chronologically.
- **Feature Engineering:** Generates moving averages, lag features, and daily returns.
- **Model Training:** Uses Random Forest Regression on historical data.
- **Model Performance:** Displays MAE, MSE, RMSE, and R² Score.
- **Visualizations:** Interactive Plotly charts for Actual vs Predicted and Feature Importance.
- **Next-Day Prediction:** Predicts tomorrow's closing price based on the latest data.

## Installation
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run app.py
```
