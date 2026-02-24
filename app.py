import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Configure the Streamlit page
st.set_page_config(
    page_title="NVIDIA H100 GPU Price Tracker",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 NVIDIA H100 GPU Price Tracker")
st.markdown("Real-time tracking of NVIDIA H100 GPU pricing trends")

# Placeholder JSON URL (replace with actual data source later)
JSON_URL = "https://api.example.com/gpu-prices/h100"

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_price_data():
    """
    Fetch price data. For now, this generates realistic sample data 
    until we connect the live API.
    """
    try:
        # Generate 30 days of sample price data
        dates = pd.date_range(start=datetime.now() - timedelta(days=29), end=datetime.now(), freq='D')
        
        # Simulate realistic H100 pricing (around $25,000-$35,000)
        random.seed(42)  # For consistent demo data
        
        sample_data = []
        base_price = 30000
        
        for i, date in enumerate(dates):
            # Add some realistic price variation
            daily_change = random.uniform(-0.05, 0.05)  # ±5% daily variation
            price = base_price * (1 + daily_change + 0.001 * i)  # Slight upward trend
            
            sample_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(price, 2),
                'volume': random.randint(10, 100)
            })
        
        return pd.DataFrame(sample_data)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
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
    """Create an interactive price trend chart"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines+markers',
        name='H100 Price',
        line=dict(color='#00cc96', width=3),
        marker=dict(size=6),
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # --- FIXED TRENDLINE MATH ---
    # Using numpy instead of scipy to prevent installation crashes
    x_numeric = np.arange(len(df))
    y_values = df['price'].values
    # Calculate linear regression (1st degree polynomial)
    z = np.polyfit(x_numeric, y_values, 1)
    p = np.poly1d(z)
    trend_line = p(x_numeric)
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=trend_line,
        mode='lines',
        name='Linear Trend Line',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='<b>Trend:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='NVIDIA H100 GPU Price Trend',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        showlegend=True,
        height=500,
        template='plotly_white'
    )
    
    # --- FIXED TYPO: Changed update_yaxis to update_yaxes ---
    fig.update_yaxes(tickformat='$,.0f')
    
    return fig

# Main dashboard
def main():
    # Fetch the data
    with st.spinner('Loading price data...'):
        df = fetch_price_data()
    
    if df.empty:
        st.error("Unable to load price data. Please check your connection and try again.")
        return
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate metrics
    current_price = df.iloc[-1]['price']
    change_amount, change_percent = calculate_price_change(df)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:,.2f}",
        )
    
    with col2:
        st.metric(
            label="24h Change",
            value=f"${change_amount:,.2f}",
            delta=f"{change_percent:.2f}%"
        )
    
    with col3:
        st.metric(
            label="30-Day High",
            value=f"${df['price'].max():,.2f}"
        )
    
    with col4:
        st.metric(
            label="30-Day Low",
            value=f"${df['price'].min():,.2f}"
        )
    
    # Price trend chart
    st.subheader("📈 Price Trend Analysis")
    fig = create_price_chart(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional statistics
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
    
    # Data source info
    st.sidebar.header("ℹ️ Information")
    st.sidebar.info(
        "This dashboard tracks NVIDIA H100 GPU prices using daily market data. "
        "Prices are updated regularly throughout the day."
    )
    
    st.sidebar.header("🔄 Data Source")
    st.sidebar.code(JSON_URL, language="text")
    st.sidebar.caption("Replace with your actual data source URL")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()
