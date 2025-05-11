import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pytz
import numpy as np
import time

# ------------------ KONFIGURASI ------------------
DAYS = 30
TIMEZONE = 'Asia/Jakarta'
COINS = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA']

# ------------------ UTIL ------------------
def convert_to_local_time(utc_time, timezone=TIMEZONE):
    local_tz = pytz.timezone(timezone)
    utc_time = pd.to_datetime(utc_time)
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)
    return utc_time.astimezone(local_tz)

# ------------------ API & MODEL ------------------
def get_current_coin_price(coin_symbol):
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        r = requests.get(url, params={'symbol': f'{coin_symbol}USDT'}, timeout=5)
        r.raise_for_status()
        return float(r.json()['price'])
    except Exception as e:
        st.warning(f"Gagal mengambil harga {coin_symbol}: {e}")
        return None

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
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        st.warning(f"Gagal ambil data historis {coin_symbol}: {e}")
        return None
    
    cols = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
            'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume', 
            'Taker_buy_quote_asset_volume', 'Ignore']
    
    df = pd.DataFrame(data, columns=cols)
    df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Close'] = pd.to_numeric(df['Close'])
    return df[['Datetime', 'Close']]

def calculate_technical_indicators(df):
    df = df.copy().sort_values('Datetime')
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi().fillna(0)
    df['SMA'] = ta.trend.SMAIndicator(df['Close'], window=14).sma_indicator().fillna(0)
    df['EMA'] = ta.trend.EMAIndicator(df['Close'], window=14).ema_indicator().fillna(0)
    df['MACD'] = ta.trend.MACD(df['Close']).macd().fillna(0)
    df['ROC'] = ta.momentum.ROCIndicator(df['Close'], window=14).roc().fillna(0)
    return df

def predict_price(coin_symbol):
    coin_df = get_coin_data_from_binance(coin_symbol, days=DAYS)
    if coin_df is None or coin_df.empty:
        return None, None, None
    
    coin_df = calculate_technical_indicators(coin_df)

    features = ['RSI', 'SMA', 'EMA', 'MACD', 'ROC']
    X = coin_df[features]
    y = coin_df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    latest_features = coin_df.iloc[-1][features].values.reshape(1, -1)
    pred_price = model.predict(latest_features)[0]

    current_price = get_current_coin_price(coin_symbol)
    if current_price is None:
        return None, None, None

    price_change_percent = ((pred_price - current_price) / current_price) * 100
    return current_price, pred_price, price_change_percent

# ------------------ STREAMLIT UI ------------------
st.title("📈 Prediksi Harga Koin Populer (1 Jam ke Depan)")

cols = st.columns(5)

for i, coin in enumerate(COINS):
    with cols[i]:
        current_price, pred_price, change = predict_price(coin)
        time.sleep(0.5)  # hindari spam API

        if current_price is None or pred_price is None:
            st.error(f"Gagal memproses {coin}")
            continue

        if change > 0:
            color = "green"
            message = f"⬆️ Naik {change:.2f}%"
        else:
            color = "red"
            message = f"⬇️ Turun {abs(change):.2f}%"

        st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; color:white; text-align:center">
            <h3>{coin}</h3>
            <p>Harga Sekarang: <b>${current_price:.2f}</b></p>
            <p>Prediksi 1 jam lagi: <b>${pred_price:.2f}</b></p>
            <p style="font-size:18px">{message}</p>
        </div>
        """, unsafe_allow_html=True)
