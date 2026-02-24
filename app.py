import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re
import os

# Configure the Streamlit page
st.set_page_config(page_title="H100 Live Rental Index", page_icon="📈", layout="wide")

st.title("📈 NVIDIA H100 Live Cloud Rental Index")
st.markdown("Real-time web scraping of hourly H100 rental rates vs. NVDA Stock")

@st.cache_data(ttl=600) # Refreshes scrape every 10 minutes
def scrape_live_h100_price():
    """
    Scrapes gpus.io for real-time H100 hourly rental prices.
    """
    url = "https://gpus.io/?sortColumn=pricePerGpu&sortDirection=asc&filter_gpuType=NVIDIA+H100"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all text that looks like a price (e.g., $2.50)
        text_content = soup.get_text()
        prices = re.findall(r'\$\s*([0-9]+\.[0-9]{2})', text_content)
        
        # Convert to floats and filter for realistic hourly rental rates ($1.00 to $10.00)
        valid_prices = [float(p) for p in prices if 1.0 <= float(p) <= 10.0]
        
        if valid_prices:
            # Return the median market rate
            return np.median(valid_prices)
        else:
            return 2.85 
            
    except Exception as e:
        st.error(f"Scraping failed: {e}")
        return 2.85

def load_historical_data():
    """Reads the permanent database updated by GitHub Actions"""
    try:
        df = pd.read_csv('prices.csv')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df
    except Exception:
        # If the file doesn't exist yet, return an empty dataframe
        return pd.DataFrame(columns=['date', 'price'])

@st.cache_data(ttl=3600)
def fetch_stock_data():
    """Fetches NVDA stock history."""
    try:
        stock = yf.Ticker("NVDA")
        hist = stock.history(period="1mo").reset_index()
        if hist['Date'].dt.tz is not None:
            hist['Date'] = hist['Date'].dt.tz_localize(None)
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        return hist[['Date', 'Close']]
    except Exception:
        return pd.DataFrame()

def main():
    # 1. Scrape the current live price for the top metric card
    with st.spinner('Checking current live market rate...'):
        live_price = scrape_live_h100_price()
        
    # 2. Load the permanent historical data from your CSV
    df = load_historical_data()
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # 3. Combine live data with historical data so the chart is perfectly up to date
    if not df.empty:
        if today_str in df['date'].values:
            df.loc[df['date'] == today_str, 'price'] = live_price
        else:
            new_row = pd.DataFrame({'date': [today_str], 'price': [live_price]})
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        # Failsafe if prices.csv hasn't been created yet
        df = pd.DataFrame({'date': [today_str], 'price': [live_price]})
    
    # Calculate metrics
    if len(df) >= 2:
        current_price = df.iloc[-1]['price']
        yesterday_price = df.iloc[-2]['price']
        change_amt = current_price - yesterday_price
        change_pct = (change_amt / yesterday_price) * 100
    else:
        current_price = live_price
        change_amt = 0.0
        change_pct = 0.0

    # Layout Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Live Market Median (Hourly)", value=f"${current_price:,.2f}/hr")
    with col2:
        st.metric(label="24h Change", value=f"${change_amt:,.2f}", delta=f"{change_pct:.2f}%")
    with col3:
        st.metric(label="Data Source Status", value="Live (gpus.io) + CSV DB")

    # Chart
    st.subheader("📊 Live Market Index vs NVDA Stock")
    stock_df = fetch_stock_data()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Left Axis: GPU Rental Price
    fig.add_trace(go.Scatter(x=df['date'], y=df['price'], mode='lines+markers', name='H100 Rental Rate ($/hr)', line=dict(color='#00cc96', width=3)), secondary_y=False)
    
    # Right Axis: NVDA Stock
    if not stock_df.empty:
        fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Close'], mode='lines', name='NVDA Stock Price', line=dict(color='#ff9900', width=2, dash='dot')), secondary_y=True)

    fig.update_layout(height=500, template='plotly_white', hovermode='x unified', legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    fig.update_yaxes(title_text="<b>Hourly Rate</b> (USD)", tickformat='$.2f', secondary_y=False)
    fig.update_yaxes(title_text="<b>NVDA Stock</b> (USD)", tickformat='$,.0f', secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
