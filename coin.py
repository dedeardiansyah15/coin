import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pytz
import numpy as np

# ------------------ KONFIGURASI ------------------
DAYS = 30  # Jumlah hari data historis yang diambil
TIMEZONE = 'Asia/Jakarta'
COINS = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA']  # Daftar 5 koin populer

# ------------------ UTIL ------------------

# Fungsi untuk mengonversi waktu UTC ke waktu lokal
def convert_to_local_time(utc_time, timezone=TIMEZONE):
    local_tz = pytz.timezone(timezone)
    if isinstance(utc_time, (np.datetime64, pd.Timestamp)):
        utc_time = pd.to_datetime(utc_time).to_pydatetime()
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    return utc_time.astimezone(local_tz)

# ------------------ FUNGSI ------------------

# Fungsi untuk mendapatkan harga koin terkini
def get_current_coin_price(coin_symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {'symbol': f'{coin_symbol}USDT'}
    r = requests.get(url, params=params)
    if r.status_code == 200:
        data = r.json()
        return float(data['price'])
    else:
        raise RuntimeError(f"Binance error {r.status_code}")

# Fungsi untuk mengambil data harga koin dari Binance (interval 6 jam)
def get_coin_data_from_binance(coin_symbol, days=30, interval='6h'):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': f'{coin_symbol}USDT',
        'interval': interval,
        'startTime': int(start.timestamp() * 1000),
        'endTime': int(end.timestamp() * 1000)
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"Binance error {r.status_code}")
    
    data = r.json()
    cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
            'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 
            'Taker_buy_quote_asset_volume', 'Ignore']
    
    df = pd.DataFrame(data, columns=cols)
    df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['date'] = df['Datetime'].dt.date
    df['hour'] = df['Datetime'].dt.hour
    df['Close'] = pd.to_numeric(df['Close'])
    return df[['Datetime', 'date', 'hour', 'Close']]

# Fungsi untuk menghitung indikator teknikal
def calculate_technical_indicators(df):
    df = df.copy().sort_values('Datetime')
    df['RSI']  = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().fillna(0)
    df['SMA']  = ta.trend.SMAIndicator(df['Close'], window=14).sma_indicator().fillna(0)
    df['EMA']  = ta.trend.EMAIndicator(df['Close'], window=14).ema_indicator().fillna(0)
    df['MACD'] = ta.trend.MACD(df['Close']).macd().fillna(0)
    df['ROC']  = ta.momentum.ROCIndicator(df['Close'], window=14).roc().fillna(0)
    return df

# Fungsi untuk membuat model prediksi dan menghitung persentase perubahan
def predict_price(coin_symbol):
    # Ambil data harga koin
    coin_df = get_coin_data_from_binance(coin_symbol, days=30, interval='6h')
    coin_df = calculate_technical_indicators(coin_df)

    # Siapkan fitur & target untuk model
    features = ['RSI', 'SMA', 'EMA', 'MACD', 'ROC']
    X = coin_df[features]
    y = coin_df['Close']

    # Split data menjadi train dan test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Latih model RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediksi harga pada data test
    y_pred = model.predict(X_test)

    # Prediksi harga 1 jam ke depan
    latest = coin_df.iloc[-1]  # Ambil data harga terbaru
    latest_features = latest[features].values.reshape(1, -1)  # Siapkan fitur untuk prediksi
    pred_price = model.predict(latest_features)[0]  # Prediksi harga 1 jam ke depan

    # Ambil harga saat ini
    current_price = get_current_coin_price(coin_symbol)

    # Hitung persentase perubahan harga
    price_change_percent = ((pred_price - current_price) / current_price) * 100

    return current_price, pred_price, price_change_percent

# ------------------ STREAMLIT INTERFACE ------------------

st.title("Prediksi Harga Koin Terpopuler")

# Membuat kotak untuk setiap koin dengan warna hijau/merah berdasarkan perubahan harga
for coin in COINS:
    current_price, pred_price, price_change_percent = predict_price(coin)
    
    # Menentukan warna kotak berdasarkan perubahan harga
    if price_change_percent > 0:
        box_color = "background-color: green; color: white;"
        direction = f"Harga diprediksi akan **naik** sebanyak {price_change_percent:.2f}%"
    else:
        box_color = "background-color: red; color: white;"
        direction = f"Harga diprediksi akan **turun** sebanyak {abs(price_change_percent):.2f}%"
    
    # Menampilkan kotak untuk setiap koin
    st.markdown(f"""
    <div style="padding: 10px; margin-bottom: 10px; border-radius: 5px; {box_color}">
        <h3>{coin}</h3>
        <p>Harga saat ini: ${current_price:.2f}</p>
        <p>Prediksi harga 1 jam ke depan: ${pred_price:.2f}</p>
        <p>{direction}</p>
    </div>
    """, unsafe_allow_html=True)
