import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")

# ==================================================
# 9. STREAMLIT DESIGN (Title and Subtitle)
# ==================================================
st.title("📈 Stock Price Prediction Using Machine Learning")
st.subheader("End-to-End Machine Learning Application")

# ==================================================
# 1. DATASET ACQUISITION (Live Fetch OR Upload)
# ==================================================
st.header("📊 Dataset Overview")

data_source = st.radio("Choose Data Source:", ("Fetch Live Data (Yahoo Finance)", "Upload Custom CSV Dataset"))

df_raw = pd.DataFrame()

if data_source == "Fetch Live Data (Yahoo Finance)":
    ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL, MSFT, TCS.NS)", "RELIANCE.NS")
    if st.button("Fetch Data & Analyze"):
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                # Fetch 2 years of daily data
                stock_data = yf.download(ticker, period="2y", interval="1d")
                if stock_data.empty:
                    st.error("No data found. Please check the ticker symbol.")
                    st.stop()
                
                # Flatten multi-index columns if present (yfinance behavior)
                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.droplevel(1)
                
                stock_data = stock_data.reset_index()
                
                # Standardize column names from yfinance
                df_raw = stock_data.rename(columns={
                    'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                })
                
                st.success(f"Successfully fetched {len(df_raw)} records for {ticker}!")
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                st.stop()

elif data_source == "Upload Custom CSV Dataset":
    uploaded_file = st.file_uploader("Upload your historical stock data CSV file", type=["csv"])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)

if not df_raw.empty:
    try:
        st.write("### Dataset Preview")
        st.dataframe(df_raw.head())
        
        col1, col2 = st.columns(2)
        col1.metric("Total Rows", df_raw.shape[0])
        col2.metric("Total Columns", df_raw.shape[1])
        
        with st.expander("Show Column Names & Missing Values"):
            missing_info = pd.DataFrame({
                'Column Name': df_raw.columns,
                'Missing Values': df_raw.isnull().sum()
            })
            st.dataframe(missing_info)

        st.write("### Column Mapping")
        
        # Auto-detect column names
        cols = df_raw.columns.tolist()
        
        def find_col(possible_names):
            for name in possible_names:
                for c in cols:
                    if name.lower() in str(c).lower():
                        return c
            return cols[0]
            
        col_date = st.selectbox("Date Column", cols, index=cols.index(find_col(['date', 'time', 'day'])))
        col_close = st.selectbox("Close (Target) Column", cols, index=cols.index(find_col(['close', 'price'])))
        col_open = st.selectbox("Open Column", cols, index=cols.index(find_col(['open'])))
        col_high = st.selectbox("High Column", cols, index=cols.index(find_col(['high'])))
        col_low = st.selectbox("Low Column", cols, index=cols.index(find_col(['low'])))
        col_vol = st.selectbox("Volume Column", cols, index=cols.index(find_col(['vol', 'volume'])))
        
        required_cols = [col_date, col_close, col_open, col_high, col_low, col_vol]
        if len(set(required_cols)) < 6:
            st.warning("Please ensure all selected columns are unique.")
            st.stop()
            
        # ==================================================
        # 2. DATA PREPROCESSING
        # ==================================================
        st.header("🧹 Data Preprocessing")
        
        with st.spinner("Preprocessing Data..."):
            df = df_raw[required_cols].copy()
            
            # Convert date
            try:
                # Check if Date is already datetime (from yfinance)
                if not pd.api.types.is_datetime64_any_dtype(df[col_date]):
                    # Check for timezone awareness in strings or convert
                    df[col_date] = pd.to_datetime(df[col_date], utc=True)
                
                # Make timezone-naive for plotly/consistency
                df[col_date] = df[col_date].dt.tz_localize(None)
            except Exception as e:
                st.error(f"Error converting Date column to datetime: {e}")
                st.stop()
                
            # Sort chronologically
            df = df.sort_values(by=col_date).reset_index(drop=True)
            
            # Numeric conversion
            numeric_cols = [col_open, col_high, col_low, col_close, col_vol]
            for c in numeric_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                
            # Handle missing and duplicates
            rows_before = len(df)
            df = df.dropna()
            df = df.drop_duplicates()
            rows_after = len(df)
            
            if rows_after < 50:
                st.error("Dataset has too few rows after cleaning. Please provide more historical data.")
                st.stop()
                
            st.success(f"Preprocessing Complete! Removed {rows_before - rows_after} invalid/missing rows.")
            st.write("Cleaned Dataset Preview:")
            st.dataframe(df.head())
            
            # Historical Plot
            st.write("### Historical Stock Price")
            fig_hist = px.line(df, x=col_date, y=col_close, title="Historical Closing Price")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ==================================================
        # 3. FEATURE ENGINEERING
        # ==================================================
        st.header("⚙️ Feature Engineering")
        
        with st.spinner("Engineering Features..."):
            df_feat = df.copy()
            
            # Previous day close
            df_feat['Prev_Close'] = df_feat[col_close].shift(1)
            
            # Lags
            df_feat['Close_Lag_2'] = df_feat[col_close].shift(2)
            df_feat['Close_Lag_3'] = df_feat[col_close].shift(3)
            df_feat['Close_Lag_5'] = df_feat[col_close].shift(5)
            df_feat['Close_Lag_7'] = df_feat[col_close].shift(7)
            
            # Moving Averages
            df_feat['MA_7'] = df_feat[col_close].rolling(window=7).mean()
            df_feat['MA_14'] = df_feat[col_close].rolling(window=14).mean()
            
            # Daily Return
            df_feat['Daily_Return'] = df_feat[col_close].pct_change()
            
            # Target: NEXT DAY'S CLOSING PRICE
            df_feat['Target_Next_Close'] = df_feat[col_close].shift(-1)
            
            # Remove rows with NaN (due to shifts and rolling windows)
            df_feat = df_feat.dropna().reset_index(drop=True)
            
            feature_cols = [
                col_open, col_high, col_low, col_close, col_vol, 
                'Prev_Close', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_5', 'Close_Lag_7', 
                'MA_7', 'MA_14', 'Daily_Return'
            ]
            target_col = 'Target_Next_Close'
            
            st.write("Created time-series features and target variable (Next Day Close).")
            st.dataframe(df_feat[feature_cols + [target_col]].head())

        # ==================================================
        # 4. TRAINING AND TESTING
        # ==================================================
        st.header("🤖 Model Training")
        
        # 80/20 Chronological Split
        split_idx = int(len(df_feat) * 0.8)
        
        train_data = df_feat.iloc[:split_idx]
        test_data = df_feat.iloc[split_idx:]
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]
        
        st.write(f"**Training data size:** {len(X_train)} rows")
        st.write(f"**Testing data size:** {len(X_test)} rows")
        
        # ==================================================
        # 5. MACHINE LEARNING MODEL
        # ==================================================
        with st.spinner("Training Random Forest Regressor..."):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test)
            
            st.success("Model trained successfully!")

        # ==================================================
        # 6. MODEL EVALUATION
        # ==================================================
        st.header("📈 Model Performance")
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("MAE", f"{mae:.2f}")
        m_col2.metric("MSE", f"{mse:.2f}")
        m_col3.metric("RMSE", f"{rmse:.2f}")
        m_col4.metric("R² Score", f"{r2:.4f}")

        # ==================================================
        # 7. VISUALIZATIONS
        # ==================================================
        st.header("📊 Actual vs Predicted")
        
        results_df = pd.DataFrame({
            'Date': test_data[col_date],
            'Actual Close': y_test,
            'Predicted Close': y_pred
        })
        
        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Actual Close'], mode='lines', name='Actual', line=dict(color='blue')))
        fig_vs.add_trace(go.Scatter(x=results_df['Date'], y=results_df['Predicted Close'], mode='lines', name='Predicted', line=dict(color='orange')))
        fig_vs.update_layout(title="Actual vs Predicted Next Day Closing Price (Testing Data)", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_vs, use_container_width=True)
        
        st.header("🔍 Feature Importance")
        importance = model.feature_importances_
        feat_imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        }).sort_values(by='Importance', ascending=True)
        
        fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', title="Random Forest Feature Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

        # ==================================================
        # 8. NEXT-DAY PREDICTION
        # ==================================================
        st.header("🔮 Predict Next Day's Stock Price")
        
        last_row_df = df.copy()
        last_row_df['Prev_Close'] = last_row_df[col_close].shift(1)
        last_row_df['Close_Lag_2'] = last_row_df[col_close].shift(2)
        last_row_df['Close_Lag_3'] = last_row_df[col_close].shift(3)
        last_row_df['Close_Lag_5'] = last_row_df[col_close].shift(5)
        last_row_df['Close_Lag_7'] = last_row_df[col_close].shift(7)
        last_row_df['MA_7'] = last_row_df[col_close].rolling(window=7).mean()
        last_row_df['MA_14'] = last_row_df[col_close].rolling(window=14).mean()
        last_row_df['Daily_Return'] = last_row_df[col_close].pct_change()
        
        # Extract the last row
        latest_data = last_row_df.iloc[-1:]
        
        if latest_data.isnull().values.any():
            st.warning("Not enough historical data to generate features for tomorrow's prediction.")
        else:
            X_latest = latest_data[feature_cols]
            latest_date = latest_data[col_date].values[0]
            last_close = latest_data[col_close].values[0]
            
            next_day_pred = model.predict(X_latest)[0]
            price_change = next_day_pred - last_close
            change_pct = (price_change / last_close) * 100
            
            st.info("⚠️ This is a machine-learning estimate based on historical data and **NOT a guaranteed future price**.")
            
            p_col1, p_col2, p_col3 = st.columns(3)
            p_col1.metric(f"Last Available Close (Date: {pd.to_datetime(str(latest_date)).strftime('%Y-%m-%d')})", f"{last_close:.2f}")
            p_col2.metric("Predicted Next Day Close", f"{next_day_pred:.2f}", f"{price_change:.2f} ({change_pct:.2f}%)")
            
    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")

else:
    st.info("Please select a data source and provide the data to get started.")
