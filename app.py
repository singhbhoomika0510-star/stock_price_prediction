"""
app.py
======
Flask Web Server for Stock Price Prediction ML Application.
Serves the modern, responsive warm-light themed web interface and provides REST APIs.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import ml_engine

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Preset popular stocks
POPULAR_STOCKS = ml_engine.POPULAR_STOCKS


@app.route('/')
def home():
    """Renders the main single-page application."""
    return render_template('index.html', popular_stocks=POPULAR_STOCKS)


@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    """Returns the list of recommended popular stocks."""
    return jsonify({
        "status": "success",
        "stocks": POPULAR_STOCKS
    })


@app.route('/api/predict', methods=['POST'])
def predict_stock():
    """
    Main API endpoint:
    Accepts JSON payload: { symbol, period, days_ahead }
    Returns full machine learning metrics, future predictions, and chart vectors.
    """
    try:
        data = request.get_json(force=True) or {}
        symbol = data.get('symbol', 'RELIANCE.NS').strip().upper()
        period = data.get('period', '1y')
        days_ahead = int(data.get('days_ahead', 7))

        if not symbol:
            return jsonify({"status": "error", "message": "Stock symbol cannot be empty."}), 400

        # Validate period
        valid_periods = ['6mo', '1y', '2y', '5y']
        if period not in valid_periods:
            period = '1y'

        # Run the full ML training & prediction pipeline
        results = ml_engine.run_full_pipeline(
            symbol=symbol,
            period=period,
            days_ahead=days_ahead
        )

        return jsonify(results)

    except Exception as e:
        app.logger.exception("Error in /api/predict")
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Returns educational documentation and Viva / Practical guide
    explaining the Linear Regression model, formulas, and evaluation.
    """
    return jsonify({
        "status": "success",
        "model_name": "Multiple Linear Regression (Ordinary Least Squares)",
        "formula": "y = β₀ + β₁·Lag₁ + β₂·Lag₂ + β₃·Lag₃ + β₄·SMA₅ + β₅·SMA₁₀ + β₆·SMA₂₀ + β₇·Volatility + β₈·VolumeChange",
        "key_concepts": [
            {
                "title": "Why Linear Regression for Stock Forecasting?",
                "description": "Linear Regression provides complete transparency and mathematical explainability without the black-box nature of deep neural networks. It calculates direct weights (coefficients) for each past price indicator."
            },
            {
                "title": "Elimination of Data Leakage (Lookahead Bias)",
                "description": "Stock prices are time-series sequences. If we use standard random train-test splitting (shuffling), future prices are used to predict past prices. In this project, we enforce a strict 80/20 chronological split (Shuffle=False) ensuring the model only learns from historical past data."
            },
            {
                "title": "Autoregressive Feature Engineering",
                "description": "We construct past price lags (Lag 1, 2, 3), short- and medium-term Simple Moving Averages (5, 10, 20-day SMAs), and normalized day volatility. The target is the Next Day Close price."
            },
            {
                "title": "Realistic Evaluation Metrics",
                "description": "We report MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and R² (Coefficient of Determination) alongside Directional Accuracy (% of days the upward/downward movement was correctly anticipated)."
            }
        ],
        "viva_questions": [
            {
                "q": "What is the loss function optimized by Linear Regression?",
                "a": "Ordinary Least Squares (OLS) minimizes the Residual Sum of Squares (RSS), which is the sum of squared differences between actual prices and predicted prices: RSS = Σ(yᵢ - ŷᵢ)²."
            },
            {
                "q": "What does an R² score of 0.85 indicate?",
                "a": "It means that 85% of the total variance in the test stock prices is explained by the linear combination of our input features (lags and moving averages)."
            },
            {
                "q": "Why is Mean Absolute Error (MAE) easier to interpret than MSE?",
                "a": "MAE is measured in the exact same units as the stock price (e.g. dollars or rupees). An MAE of 4.5 means predictions deviate from actual price by an average of 4.5 units."
            },
            {
                "q": "Can any machine learning model guarantee 100% stock market predictions?",
                "a": "No. Stock prices are influenced by unforeseen real-world events, company earnings surprises, macroeconomic shifts, and human sentiment (Efficient Market Hypothesis). ML predictions are probabilistic estimations for trend analysis, not financial guarantees."
            }
        ]
    })


if __name__ == '__main__':
    # Run locally on port 5000
    print("Starting Stock Price Prediction Web App on http://127.0.0.1:5000 ...")
    app.run(debug=True, host='0.0.0.0', port=5000)
