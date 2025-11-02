import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import datetime
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib # For saving/loading scaler

# --- Configuration ---
MODEL_FILENAME = 'stock_prediction_model.keras'
SCALER_FILENAME = 'stock_scaler.pkl'

STOCKS = {
    'Google (GOOGL)': 'GOOGL',
    'Apple (AAPL)': 'AAPL',
    'Microsoft (MSFT)': 'MSFT',
    'Amazon (AMZN)': 'AMZN',
    'Tesla (TSLA)': 'TSLA',
    'NVDIA (NVDA)': 'NVDA',
    'Reliance (RELIANCE.NS)': 'RELIANCE.NS',
    'TCS (TCS.NS)': 'TCS.NS',
    'Tata Motors (TATAMOTORS.NS)': 'TATAMOTORS.NS'
}

DEFAULT_STOCK_NAME = 'Google (GOOGL)'
DEFAULT_STOCK_SYMBOL = STOCKS[DEFAULT_STOCK_NAME]

START_DATE = '2015-01-01'
END_DATE = '2026-01-01' # Keep this as a general end date for historical data
TRAINING_PERCENTAGE = 0.80
WINDOW_SIZE_MAIN_MODEL = 50
EPOCHS_MAIN_MODEL = 100
N_FUTURE_DAYS = 7

# --- Streamlit App Title and Description ---
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Predictor")
st.write("Predict future stock prices using an LSTM deep learning model.")

# --- Data Fetching Function ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(symbol, start_date, end_date):
    """Fetches historical stock data using yfinance."""
    st.info(f"Fetching data for: **{symbol}** from **{start_date}** to **{end_date}**...")
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        st.error(f"No data found for {symbol}. Check the symbol and date range.")
        return pd.DataFrame()
    data.reset_index(inplace=True)
    data.dropna(inplace=True)
    st.success("Data fetched successfully!")
    st.dataframe(data.tail())
    return data

# --- Model Training and Saving Function ---
# Use st.cache_resource for models and heavy objects
@st.cache_resource
def train_and_save_model(data_df, model_filename, scaler_filename):
    """Trains the LSTM model and saves it with early stopping and learning rate reduction."""
    st.subheader("Training Main Model with Callbacks...")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_df[['Close']])

    train_size = int(len(data_df) * TRAINING_PERCENTAGE)
    data_train_scale = scaled_data[0:train_size]
    data_val_scale = scaled_data[train_size:]

    x_train = []
    y_train = []
    for i in range(WINDOW_SIZE_MAIN_MODEL, data_train_scale.shape[0]):
        x_train.append(data_train_scale[i - WINDOW_SIZE_MAIN_MODEL:i])
        y_train.append(data_train_scale[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_val = []
    y_val = []
    if len(data_val_scale) > WINDOW_SIZE_MAIN_MODEL:
        for i in range(WINDOW_SIZE_MAIN_MODEL, data_val_scale.shape[0]):
            x_val.append(data_val_scale[i - WINDOW_SIZE_MAIN_MODEL:i])
            y_val.append(data_val_scale[i, 0])
        x_val, y_val = np.array(x_val), np.array(y_val)
    else:
        st.warning("Not enough data for a proper validation set for the main model. Training without validation.")
        x_val, y_val = None, None

    model = Sequential()
    model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=120, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(units=150, activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.000005,
        verbose=1
    )

    callbacks = [early_stopping, reduce_lr]

    if x_val is not None and y_val is not None and len(x_val) > 0:
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS_MAIN_MODEL,
            batch_size=64,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=0 # Suppress verbose output in Streamlit
        )
    else:
        history = model.fit(
            x_train, y_train,
            epochs=EPOCHS_MAIN_MODEL,
            batch_size=64,
            verbose=0 # Suppress verbose output in Streamlit
        )

    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    st.success(f"Model saved as '{model_filename}' and Scaler saved as '{scaler_filename}'")

    return model, scaler

# --- Model Evaluation Function ---
def evaluate_model(model, data_df, scaler):
    """Evaluates the main LSTM model and plots results."""
    st.subheader("Model Evaluation")
    
    scaled_data = scaler.transform(data_df[['Close']])

    if len(scaled_data) < WINDOW_SIZE_MAIN_MODEL:
        st.warning("Not enough historical data to perform full evaluation with current window size.")
        return

    x_test = []
    y_test = []
    
    test_start_index = int(len(data_df) * TRAINING_PERCENTAGE)
    
    relevant_data_for_test = scaled_data[max(0, test_start_index - WINDOW_SIZE_MAIN_MODEL):]

    if len(relevant_data_for_test) < WINDOW_SIZE_MAIN_MODEL:
        st.warning("Not enough data to create test sequences with the specified window size. Evaluation skipped.")
        return

    for i in range(WINDOW_SIZE_MAIN_MODEL, len(relevant_data_for_test)):
        x_test.append(relevant_data_for_test[i - WINDOW_SIZE_MAIN_MODEL:i])
        y_test.append(relevant_data_for_test[i, 0])
    
    x_test, y_test = np.array(x_test), np.array(y_test)

    if len(x_test) == 0:
        st.warning("No test sequences could be created. Evaluation skipped.")
        return

    y_predict = model.predict(x_test, verbose=0)

    y_predict = scaler.inverse_transform(y_predict)
    y = scaler.inverse_transform(y_test.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y, 'g', label="Original Price")
    ax.plot(y_predict, 'r', label="Predicted Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title("Original vs. Predicted Stock Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    mae = mean_absolute_error(y, y_predict)
    rmse = np.sqrt(mean_squared_error(y, y_predict))
    r2 = r2_score(y, y_predict)
    accuracy_percentage = r2 * 100

    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R-squared (R2):** {r2:.4f}")
    st.write(f"**Accuracy Percentage:** {accuracy_percentage:.2f}%")

# --- 7-Day Prediction Function (using the main model) ---
def predict_7_day_price(main_model, data_df, symbol, scaler):
    """Uses the main model for 7-day iterative prediction and plots."""
    st.subheader(f"7-Day Price Prediction using Main Model for {symbol}")

    data_df['Date'] = pd.to_datetime(data_df['Date'])
    df = data_df.set_index('Date')[['Close']]

    df_scaled_full = scaler.transform(df)

    last_window = df_scaled_full[-WINDOW_SIZE_MAIN_MODEL:].reshape(1, WINDOW_SIZE_MAIN_MODEL, 1)
    future_predictions = []

    for _ in range(N_FUTURE_DAYS):
        next_pred = main_model.predict(last_window, verbose=0)[0][0]
        future_predictions.append(next_pred)
        last_window = np.append(last_window[:, 1:, :], [[[next_pred]]], axis=1)

    future_predictions_actual = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    last_historical_date = df.index[-1]
    # Use BDay to get business days
    future_dates = pd.bdate_range(start=last_historical_date + pd.Timedelta(days=1), periods=N_FUTURE_DAYS).tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index[-2 * WINDOW_SIZE_MAIN_MODEL:], df.Close[-2 * WINDOW_SIZE_MAIN_MODEL:], label='Historical')
    ax.plot(future_dates, future_predictions_actual, label='Forecast (Business Days)')
    ax.legend()
    ax.set_title(f'Stock Price Forecast for Next {N_FUTURE_DAYS} Trading Days ({symbol})')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)
    st.pyplot(fig)
    
    return future_predictions_actual, future_dates

# --- Live Prediction Function (using the main model) ---
def live_predict_future_price(symbol, main_model, main_scaler):
    """Fetches live data and predicts future prices using the main model."""
    st.subheader(f"Live {N_FUTURE_DAYS}-Day Price Prediction for {symbol}")
    end_date = datetime.datetime.now().date()
    
    start_date = end_date - datetime.timedelta(days=WINDOW_SIZE_MAIN_MODEL + 80)

    live_data = yf.download(symbol, start=start_date, end=end_date)
    if live_data.empty:
        st.error(f"Could not fetch live data for {symbol}. Skipping live prediction.")
        return None, None
    live_data.reset_index(inplace=True)
    live_data.set_index('Date', inplace=True)
    live_data.dropna(inplace=True)

    live_df = live_data[['Close']]
    
    live_scaled = main_scaler.transform(live_df)

    if len(live_scaled) < WINDOW_SIZE_MAIN_MODEL:
        st.warning(f"Not enough recent data ({len(live_scaled)} points) for a {WINDOW_SIZE_MAIN_MODEL}-day window for live prediction. Skipping.")
        st.info(f"Please ensure at least {WINDOW_SIZE_MAIN_MODEL} trading days are fetched. Try increasing the 'days' in datetime.timedelta() in live_predict_future_price function.")
        return None, None

    last_window_live = live_scaled[-WINDOW_SIZE_MAIN_MODEL:].reshape(1, WINDOW_SIZE_MAIN_MODEL, 1)
    live_predictions = []

    for _ in range(N_FUTURE_DAYS):
        next_pred = main_model.predict(last_window_live, verbose=0)[0][0]
        live_predictions.append(next_pred)
        last_window_live = np.append(last_window_live[:, 1:, :], [[[next_pred]]], axis=1)

    live_predictions_actual = main_scaler.inverse_transform(np.array(live_predictions).reshape(-1, 1))

    last_live_date = live_df.index[-1]
    live_future_dates = pd.bdate_range(start=last_live_date + pd.Timedelta(days=1), periods=N_FUTURE_DAYS).tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(live_df.index[-WINDOW_SIZE_MAIN_MODEL:], live_df.Close[-WINDOW_SIZE_MAIN_MODEL:], label='Recent Actual')
    ax.plot(live_future_dates, live_predictions_actual, label='Live Forecast (Business Days)')
    ax.set_title(f'Live Forecast for {symbol} Stock')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.write("Live 7-Day Predictions:")
    for date, price in zip(live_future_dates, live_predictions_actual.flatten()):
        st.write(f" 🗓️ **{date.date()}**: ₹{price:.2f}")

    return live_predictions_actual, live_future_dates

# --- Main Streamlit App Logic ---

# Sidebar for stock selection
st.sidebar.header("Configuration")
selected_stock_name = st.sidebar.selectbox(
    "Select a Stock:",
    list(STOCKS.keys()),
    index=list(STOCKS.keys()).index(DEFAULT_STOCK_NAME)
)
selected_stock_symbol = STOCKS[selected_stock_name]

# Add a button to trigger re-training
retrain_model = st.sidebar.button("Retrain Model (if data updated or for fresh training)")

data_load_state = st.info("Loading data...")
data = get_stock_data(selected_stock_symbol, START_DATE, END_DATE)
data_load_state.empty()

main_model = None
main_scaler = MinMaxScaler(feature_range=(0, 1)) # Initialize for type consistency

if not data.empty:
    # Always fit scaler on the full dataset if training a new model or initially
    main_scaler.fit(data[['Close']])

    if retrain_model or not (os.path.exists(MODEL_FILENAME) and os.path.exists(SCALER_FILENAME)):
        with st.spinner("Training model... This might take a while."):
            main_model, main_scaler = train_and_save_model(data, MODEL_FILENAME, SCALER_FILENAME)
    else:
        st.info(f"Loading existing model from '{MODEL_FILENAME}' and scaler from '{SCALER_FILENAME}'...")
        try:
            main_model = load_model(MODEL_FILENAME)
            main_scaler = joblib.load(SCALER_FILENAME)
            st.success("Model and Scaler loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model or scaler: {e}. Retraining model.")
            with st.spinner("Retraining model due to load error..."):
                main_model, main_scaler = train_and_save_model(data, MODEL_FILENAME, SCALER_FILENAME)
else:
    st.warning("No data available to train or load the model. Please check stock symbol or date range.")

# Main content of the app
if main_model is not None and not data.empty:
    st.markdown("---")
    evaluate_model(main_model, data, main_scaler)
    st.markdown("---")
    
    # 7-day prediction for historical context
    future_prices, future_dates = predict_7_day_price(main_model, data, selected_stock_symbol, main_scaler)
    st.markdown("---")
    
    # Live 7-day prediction
    live_predictions_actual, live_future_dates = live_predict_future_price(
        selected_stock_symbol, main_model, main_scaler
    )
    
    st.markdown("---")
    st.subheader("Interactive 7-Day Forecast (Console Output)")
    if live_predictions_actual is not None and live_future_dates is not None:
        future_data_console = pd.DataFrame({
            "Date": live_future_dates,
            "Predicted Price": live_predictions_actual.flatten()
        })
        st.dataframe(future_data_console.set_index("Date")) # Display as a table
        
        if not future_data_console.empty:
            st.write(f"\nDefault displayed prediction (first day):")
            selected_row = future_data_console.iloc[0]
            st.success(f"Prediction for **{selected_row['Date'].date()}**: ₹**{selected_row['Predicted Price']:.2f}**")
    else:
        st.info("No future predictions available to display.")
            