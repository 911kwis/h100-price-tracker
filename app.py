import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import yfinance as yf
import random

# Configure the Streamlit page
st.set_page_config(
    page_title="NVIDIA H100 GPU Price Tracker",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 NVIDIA H100 GPU Price Tracker")
st.markdown("Real-time tracking of NVIDIA H100 GPU pricing trends vs. NVDA Stock")

# The live data source URL
JSON_URL = "https://raw.githubusercontent.com/United-Compute/gpu-price-tracker/main/data/NVIDIA_H100_PCIe_80_GB.json"

@st.cache_data(ttl=3600)
def fetch_price_data():
    """Attempts to fetch live data from the JSON API with a simulated fallback."""
    try:
        response = requests.get(JSON_URL, timeout=5)
        if response.status_code == 200:
            raw_data = response.json()
            df = pd.DataFrame(raw_data)
            df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
            df = df[['date', 'price']]
            df = df.groupby('date', as_index=False).mean()
            return df
        else:
            raise ValueError(f"API returned status code {response.status_code}")

    except Exception:
        st.warning("⚠️ Live API endpoint not reachable. Displaying simulated market data.")
        dates = pd.date_range(start=datetime.now() - timedelta(days=29), end=datetime.now(), freq='D')
        random.seed(42) 
        sample_data = []
        base_price = 30000
        for i, date in enumerate(dates):
            daily_change = random.uniform(-0.05, 0.05)
            price = base_price * (1 + daily_change + 0.001 * i)
            sample_data.append({'date': date.strftime('%Y-%m-%d'), 'price': round(price, 2)})
        return pd.DataFrame(sample_data)

@st.cache_data(ttl=3600)
def fetch_stock_data():
    """Fetches NVDA stock history for the last 6 months"""
    try:
        stock = yf.Ticker("NVDA")
        hist = stock.history(period="6mo")
        hist = hist.reset_index()
        hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
        return hist[['Date', 'Close']]
    except Exception:
        return pd.DataFrame()

def calculate_price_change(df):
    """Calculate price change from yesterday"""
    if len(df) < 2:
        return 0, 0
    today_price = df.iloc[-1]['price']
    yesterday_price = df.iloc[-2]['price']
    change_amount = today_price - yesterday_price
    change_percent = (change_amount / yesterday_price) * 100
    return change_amount, change_percent

def create_price_chart(df):
    """Create a dual-axis chart: H100 Price vs NVDA Stock"""
    stock_df = fetch_stock_data()
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Plot H100 Price (Left Axis)
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['price'],
            mode='lines+markers',
            name='H100 GPU Price',
            line=dict(color='#00cc96', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Date:</b> %{x}<br><b>GPU Price:</b> $%{y:,.0f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Plot NVDA Stock (Right Axis)
    if not stock_df.empty:
        fig.add_trace(
            go.Scatter(
                x=stock_df['Date'],
                y=stock_df['Close'],
                mode='lines',
                name='NVDA Stock Price',
                line=dict(color='#ff9900', width=2, dash='dot'),
                hovertemplate='<b>Stock Price:</b> $%{y:,.2f}<extra></extra>'
            ),
            secondary_y=True
        )
        
    # Add Trendline for H100 using numpy
    x_numeric = np.arange(len(df))
    y_values = df['price'].values
    z = np.polyfit(x_numeric, y_values, 1)
    p = np.poly1d(z)
    trend_line = p(x_numeric)
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=trend_line,
            mode='lines',
            name='H100 Trend Line',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Trend:</b> $%{y:,.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    fig.update_layout(
        title='H100 Market Price vs. NVIDIA Stock Performance',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
    )
    
    fig.update_yaxes(title_text="<b>H100 Price</b> (USD)", tickformat='$,.0f', secondary_y=False)
    fig.update_yaxes(title_text="<b>NVDA Stock</b> (USD)", tickformat='$,.0f', secondary_y=True)
    
    return fig

def main():
    with st.spinner('Loading price data...'):
        df = fetch_price_data()
    
    if df.empty:
        st.error("Unable to load price data. Please check your connection and try again.")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    current_price = df.iloc[-1]['price']
    change_amount, change_percent = calculate_price_change(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Current H100 Price", value=f"${current_price:,.2f}")
    with col2:
        st.metric(label="24h Change", value=f"${change_amount:,.2f}", delta=f"{change_percent:.2f}%")
    with col3:
        st.metric(label="30-Day High", value=f"${df['price'].max():,.2f}")
    with col4:
        st.metric(label="30-Day Low", value=f"${df['price'].min():,.2f}")
    
    st.subheader("📈 Price Trend Analysis")
    fig = create_price_chart(df)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Price Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Average Price', 'Median Price', 'Standard Deviation', 'Price Range'],
            'Value': [
                f"${df['price'].mean():,.2f}",
                f"${df['price'].median():,.2f}",
                f"${df['price'].std():,.2f}",
                f"${df['price'].max() - df['price'].min():,.2f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True)
    
    with col2:
        st.subheader("📅 Recent Price History")
        recent_data = df.tail(10)[['date', 'price']].copy()
        recent_data['date'] = recent_data['date'].dt.strftime('%Y-%m-%d')
        recent_data['price'] = recent_data['price'].apply(lambda x: f"${x:,.2f}")
        recent_data.columns = ['Date', 'Price']
        st.dataframe(recent_data, hide_index=True)
    
    st.sidebar.header("ℹ️ Information")
    st.sidebar.info("This dashboard tracks NVIDIA H100 GPU prices using market data and compares them against NVDA stock.")
    st.sidebar.header("🔄 Data Source")
    st.sidebar.code(JSON_URL, language="text")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()
