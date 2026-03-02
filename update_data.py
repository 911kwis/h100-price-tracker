import pandas as pd
import requests
import re

def fetch_api_data(tab):
    url = f"https://portal.silicondata.com/gpu-index-chart?mainTab={tab}&gpu=h100&_rsc=12tua"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        text = response.text
        
        # Look for the data patterns like: "2026-02-28", "value": 7.44
        # This regex is more flexible to handle the RSC format
        dates = re.findall(r'(\d{4}-\d{2}-\d{2})', text)
        # Find numbers that look like prices (e.g., 7.44 or 2.90) 
        # specifically appearing near the dates
        prices = re.findall(r'(?<=[:,\s])(\d\.\d{2})(?=[,\s\}])', text)
        
        # Log to GitHub console for debugging
        print(f"Tab {tab}: Found {len(dates)} dates and {len(prices)} prices.")
        
        if not dates or not prices:
            return {}

        # Zip them together (keeping the last 30 pairs)
        data_map = {}
        for i in range(min(len(dates), len(prices))):
            data_map[dates[i]] = float(prices[i])
        return data_map
        
    except Exception as e:
        print(f"Error fetching {tab}: {e}")
        return {}

def main():
    hyper_map = fetch_api_data('hyperscaler')
    neo_map = fetch_api_data('neo-cloud')
    
    if not hyper_map and not neo_map:
        print("CRITICAL: No data found. The API format might have changed.")
        return

    # Use all unique dates found
    all_dates = sorted(list(set(hyper_map.keys()) | set(neo_map.keys())))
    
    rows = []
    # Starting fallbacks in case data is sparse
    h_val, n_val = 7.44, 2.90 
    
    for d in all_dates:
        h_val = hyper_map.get(d, h_val)
        n_val = neo_map.get(d, n_val)
        rows.append({'date': d, 'sd_hyperscaler': h_val, 'sd_neocloud': n_val})
    
    df = pd.DataFrame(rows)
    df.to_csv('prices.csv', index=False)
    print(f"Success! Database updated with {len(df)} rows.")

if __name__ == "__main__":
    main()
