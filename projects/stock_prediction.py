import streamlit as st
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import plotly.graph_objs as go
import pandas as pd


# Function to get historical stock data using yfinance
def get_stock_data(stock_code):
    stock = yf.Ticker(stock_code)
    data = stock.history(period="max")
    return data


# Function to predict stock prices using different regressors
def predict_prices(data):
    # Feature engineering: Using 'Close' prices as the target variable
    data['Prediction'] = data['Close'].shift(-1)

    # Drop rows with NaN values
    data = data.dropna()

    # Separating the feature set and the target variable
    X = data.drop(['Prediction'], axis=1)
    y = data['Prediction']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# Function to train SVM Regressor
def train_svm(X_train, y_train):
    svm_regressor = SVR(kernel='rbf')
    svm_regressor.fit(X_train, y_train)
    return svm_regressor


# Function to train KNN Regressor
def train_knn(X_train, y_train):
    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(X_train, y_train)
    return knn_regressor


# Function to train XGBoost Regressor
def train_xgb(X_train, y_train):
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                     max_depth=5, alpha=10, n_estimators=100)
    xgb_regressor.fit(X_train, y_train)
    return xgb_regressor


# Function to plot historical data
def plot_historical_data(data):
    fig_historical = go.Figure()
    fig_historical.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
    fig_historical.update_layout(title='Historical Stock Prices', xaxis_title='Date', yaxis_title='Stock Price')
    st.plotly_chart(fig_historical)


# Function to plot forecasted data
def plot_forecasted_data(data, predictions):
    fig_forecast = go.Figure()
    st.write(f"Number of predictions: {len(predictions)}")
    if len(predictions) >= 1 and len(predictions[0]) > 0:
        fig_forecast.add_trace(go.Scatter(x=data.index[-len(predictions[0]):], y=predictions[0], mode='lines', name=f'{selected_model} Forecast'))
        fig_forecast.update_layout(title='Stock Price Forecast', xaxis_title='Date', yaxis_title='Stock Price')
        st.plotly_chart(fig_forecast)
    else:
        st.write("Insufficient data for plotting forecasts")


# Streamlit App
st.title('Stock Price Prediction App')

# Sidebar for user inputs
st.sidebar.title('User Inputs')
stock_code = st.sidebar.text_input('Enter Stock Code (e.g., AAPL):')

selected_model = st.sidebar.radio('Select Model', ('SVM', 'KNN', 'XGBoost'))

if stock_code:
    try:
        data = get_stock_data(stock_code)
        st.write('Historical Stock Data:')
        st.write(data.head())

        X_train, X_test, y_train, y_test = predict_prices(data)

        st.write('Data Split Successfully!')

        if selected_model == 'SVM':
            svm_model = train_svm(X_train, y_train)
            st.write('SVM Model Trained Successfully!')

        elif selected_model == 'KNN':
            knn_model = train_knn(X_train, y_train)
            st.write('KNN Model Trained Successfully!')

        elif selected_model == 'XGBoost':
            xgb_model = train_xgb(X_train, y_train)
            st.write('XGBoost Model Trained Successfully!')

        if st.button('Show Forecast'):
            try:
                if selected_model == 'SVM':
                    svm_pred = svm_model.predict(X_test)
                    predictions = [svm_pred]
                elif selected_model == 'KNN':
                    knn_pred = knn_model.predict(X_test)
                    predictions = [knn_pred]
                elif selected_model == 'XGBoost':
                    xgb_pred = xgb_model.predict(X_test)
                    predictions = [xgb_pred]

                plot_historical_data(data)
                plot_forecasted_data(data, predictions)

            except Exception as plot_error:
                st.write('Error generating plot:', plot_error)
    except Exception as e:
        st.write('Error:', e)
