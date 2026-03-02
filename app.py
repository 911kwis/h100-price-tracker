import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Configure the Streamlit page
st.set_page_config(page_title="Silicon Data H100 Index", page_icon="📈", layout="wide")

st.title("📈 Enterprise H100 Pricing Index")
st.markdown("Tracking the **Silicon Data Index** for Hyperscaler and Neo-Cloud H100 rental rates vs. NVDA stock.")

def load_historical_data():
    """Reads the permanent database updated nightly via GitHub Actions"""
    try:
        df = pd.read_csv('prices.csv')
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        return df
    except Exception:
        return pd.DataFrame(columns=['date', 'sd_hyperscaler', 'sd_neocloud'])

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
    df = load_historical_data()
    
    if len(df) < 2:
        st.warning("Database initializing...")
        return
        
    curr_hyper = df.iloc[-1]['sd_hyperscaler']
    prev_hyper = df.iloc[-2]['sd_hyperscaler']
    change_hyper = curr_hyper - prev_hyper
    
    curr_neo = df.iloc[-1]['sd_neocloud']
    prev_neo = df.iloc[-2]['sd_neocloud']
    change_neo = curr_neo - prev_neo

    spread = curr_hyper - curr_neo

    # Layout Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Hyperscaler H100 (Silicon Data)", value=f"${curr_hyper:,.2f}/hr", delta=f"${change_hyper:,.2f}")
    with col2:
        st.metric(label="Neo-Cloud H100 (Silicon Data)", value=f"${curr_neo:,.2f}/hr", delta=f"${change_neo:,.2f}")
    with col3:
        st.metric(label="Market Spread (Price Gap)", value=f"${spread:,.2f}/hr", delta_color="off")

    # Chart
    st.subheader("📊 Silicon Data Market Tiers vs NVDA Stock")
    stock_df = fetch_stock_data()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Left Axis: Hyperscaler Price (Red Line)
    fig.add_trace(go.Scatter(x=df['date'], y=df['sd_hyperscaler'], mode='lines+markers', name='Hyperscaler Rate ($/hr)', line=dict(color='#ff3366', width=3)), secondary_y=False)
    
    # Left Axis: Neo-Cloud Price (Green Line)
    fig.add_trace(go.Scatter(x=df['date'], y=df['sd_neocloud'], mode='lines+markers', name='Neo-Cloud Rate ($/hr)', line=dict(color='#00cc96', width=3)), secondary_y=False)
    
    # Right Axis: NVDA Stock (Orange Dotted Line)
    if not stock_df.empty:
        fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Close'], mode='lines', name='NVDA Stock Price', line=dict(color='#ff9900', width=2, dash='dot')), secondary_y=True)

    fig.update_layout(height=550, template='plotly_white', hovermode='x unified', legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
    fig.update_yaxes(title_text="<b>Silicon Data Rate</b> (USD)", tickformat='$.2f', secondary_y=False)
    fig.update_yaxes(title_text="<b>NVDA Stock</b> (USD)", tickformat='$,.0f', secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

    st.sidebar.info("Source: 100% Silicon Data Index. Nightly automated scraping.")

if __name__ == "__main__":
    main()
