import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objs as go
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Function for the Home page
def home_page():
    st.title('Welcome to Stock Forecasting App')
    st.text("This app helps you forecast stock prices using different regression models.")
    st.text('The regressors are SVR, KNN, XGBoost, and ARIMA.')
    st.success("Choose 'Stock Forecast' in the sidebar to start forecasting!")

# Function for the Stock Forecast page
def stock_forecast_page():
    # Sidebar title and user input for stock code
    st.sidebar.title('Stock Forecasting App')
    stock_code = st.sidebar.text_input('Enter Stock Code (e.g., AAPL):', 'AAPL')

    # Fetch stock data
    stock_data = yf.download(stock_code, start="2010-01-01", end="2023-12-31")

    # Check if stock data is not empty
    if not stock_data.empty:
        # Sidebar for model selection
        st.sidebar.subheader('Select Model')
        selected_model = st.sidebar.selectbox('Choose Model:', ('SVR', 'KNN', 'XGBoost', 'ARIMA'))

        # Prepare data for forecasting
        forecast_days = 30
        forecast_data = stock_data['Close']

        if len(stock_data) >= forecast_days:
            forecast_index = pd.date_range(start=stock_data.index[-1], periods=forecast_days, freq='B')

            if selected_model == 'SVR':
                model = SVR(kernel='rbf')
            elif selected_model == 'KNN':
                model = KNeighborsRegressor()
            elif selected_model == 'XGBoost':
                model = XGBRegressor()
            else:
                # For ARIMA
                model = None  # Placeholder for ARIMA model

            # Prepare data for ARIMA
            train_data = stock_data['Close']

            # Fit ARIMA model
            if selected_model == 'ARIMA':
                model = ARIMA(train_data, order=(5, 1, 0))  # Example ARIMA order
                model_fit = model.fit()

                # Make predictions for future dates
                forecast_values = model_fit.forecast(steps=forecast_days)
                forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast_values})
                forecast_df.set_index('Date', inplace=True)

                # Display historical data
                st.subheader('Historical Stock Data')
                st.write(stock_data)

                # Display historical and forecasted graphs
                st.subheader('Historical vs Forecasted Stock Prices')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Data'))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecasted Data'))
                fig.update_layout(title='Historical vs Forecasted Stock Prices', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig)

                # Evaluation Metrics
                actual_values = stock_data['Close'].values[-forecast_days:]

                # Calculate evaluation metrics
                mae = mean_absolute_error(actual_values, forecast_values)
                mse = mean_squared_error(actual_values, forecast_values)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual_values, forecast_values)

                st.subheader('Evaluation Metrics')
                st.write(f"Mean Absolute Error (MAE): {mae}")
                st.write(f"Mean Squared Error (MSE): {mse}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse}")
                st.write(f"R2 score: {r2}")

            else:
                # For other models (SVR, KNN, XGBoost)
                # Standard scaling for the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(forecast_data.values.reshape(-1, 1))

                X_train = scaled_data[:-forecast_days]
                y_train = stock_data['Close'].shift(-forecast_days).values[:-forecast_days]

                if len(X_train) == len(y_train):
                    model.fit(X_train, y_train)

                    forecast_values = model.predict(scaled_data[-forecast_days:])
                    forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast_values})
                    forecast_df.set_index('Date', inplace=True)

                    # Display historical data
                    st.subheader('Historical Stock Data')
                    st.write(stock_data)

                    # Display historical and forecasted graphs
                    st.subheader('Historical vs Forecasted Stock Prices')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Data'))
                    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecasted Data'))
                    fig.update_layout(title='Historical vs Forecasted Stock Prices', xaxis_title='Date', yaxis_title='Price')
                    st.plotly_chart(fig)

                    # Evaluation Metrics
                    actual_values = stock_data['Close'].values[-forecast_days:]

                    # Calculate evaluation metrics
                    mae = mean_absolute_error(actual_values, forecast_values)
                    mse = mean_squared_error(actual_values, forecast_values)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(actual_values, forecast_values)

                    st.subheader('Evaluation Metrics')
                    st.write(f"Mean Absolute Error (MAE): {mae}")
                    st.write(f"Mean Squared Error (MSE): {mse}")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
                    st.write(f"R2 score: {r2}")

                else:
                    st.warning("Inconsistent lengths of training data. Please select a different date range or adjust the forecast days.")
        else:
            st.warning("Insufficient data for forecasting. Please select a different date range.")
    else:
        st.error("No data found for the given stock code. Please enter a valid stock code.")

# Main program
def main():
    st.sidebar.title('Navigation')
    navigation = st.sidebar.radio('Go to:', ('Home', 'Stock Forecast'))

    if navigation == 'Home':
        home_page()
    else:
        stock_forecast_page()

if __name__ == "__main__":
    main()
