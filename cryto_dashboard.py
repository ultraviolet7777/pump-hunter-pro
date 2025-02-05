import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tweepy

# Load environment variables
load_dotenv()

# API Keys from .env
CRYPTOCOMPARE_API_KEY = os.getenv('CRYPTOCOMPARE_API_KEY')
KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY')
KRAKEN_API_SECRET = os.getenv('KRAKEN_API_SECRET')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
SANTIMENT_API_KEY = os.getenv('SANTIMENT_API_KEY')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')

# Twitter Authentication
auth = tweepy.OAuth1UserHandler(
    TWITTER_API_KEY, TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET
)
twitter_api = tweepy.API(auth)

# Streamlit Dashboard Setup
st.title("🚀 Pump Hunter Pro Dashboard")
st.markdown("Tracking trending tokens, predicting pumps, and sending real-time alerts! 💰")

# Adjustable Settings
refresh_interval = st.slider("Refresh Interval (seconds):", 60, 300, 120)
investment_amount = st.number_input("Investment Amount ($):", min_value=10, value=500, step=50)
momentum_threshold = st.slider("Momentum Threshold for Alerts:", 50, 100, 85)

# Placeholder for live data
placeholder = st.empty()

# Function to Fetch Data from CryptoCompare
def fetch_crypto_data():
    url = f'https://min-api.cryptocompare.com/data/top/totalvolfull?limit=10&tsym=USD&api_key={CRYPTOCOMPARE_API_KEY}'
    response = requests.get(url)
    data = response.json()
    tokens = []
    for item in data['Data']:
        tokens.append({
            'Name': item['CoinInfo']['Name'],
            'Price ($)': item['RAW']['USD']['PRICE'],
            '24h Change (%)': item['RAW']['USD']['CHANGEPCT24HOUR'],
            'Volume (24h)': item['RAW']['USD']['TOTALVOLUME24H'],
            'Market Cap': item['RAW']['USD']['MKTCAP']
        })
    return pd.DataFrame(tokens)

# Function to Send Slack Alerts
def send_slack_alert(token, score, potential_profit):
    message = {
        "text": f"🚀 *POTENTIAL PUMP ALERT!* 🚀\n\n*Token:* {token['Name']}\n*Momentum Score:* {score}/100\n*24h Change:* {token['24h Change (%)']}%\n*Price:* ${token['Price ($)']:,}\n*Potential Profit (on ${investment_amount}):* ${potential_profit:,.2f}\n\n*Act Fast!* ⚡"
    }
    requests.post(SLACK_WEBHOOK_URL, json=message)

# Basic ML Model for Pump Prediction
def predict_pump(df):
    df['Label'] = (df['24h Change (%)'] > momentum_threshold).astype(int)
    features = df[['Price ($)', '24h Change (%)', 'Volume (24h)', 'Market Cap']]
    labels = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy * 100

# Main Dashboard Loop
while True:
    with placeholder.container():
        df = fetch_crypto_data()
        if not df.empty:
            df['Potential Profit ($)'] = investment_amount * (df['24h Change (%)'] / 100)
            pump_accuracy = predict_pump(df)
            
            st.subheader(f"📈 Pump Prediction Accuracy: {pump_accuracy:.2f}%")
            st.dataframe(df.sort_values(by='24h Change (%)', ascending=False), height=400)

            # Send Slack Alerts for High Momentum Tokens
            for _, token in df.iterrows():
                if token['24h Change (%)'] >= momentum_threshold:
                    send_slack_alert(token, token['24h Change (%)'], token['Potential Profit ($)'])
        else:
            st.warning("No trending tokens found at the moment. Waiting for new data...")

    time.sleep(refresh_interval)
