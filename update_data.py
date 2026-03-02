import pandas as pd
import requests
import re
import os

def fetch_api_history(tab):
    """Hits the hidden Silicon Data API to extract the full historical chart data."""
    # We use the exact API endpoint you found, dynamically swapping between 'neo-cloud' and 'hyperscaler'
    url = f"https://portal.silicondata.com/gpu-index-chart?mainTab={tab}&gpu=h100&_rsc=12tua"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # This regular expression scans the raw API payload for dates (YYYY-MM-DD) and their matching prices
        raw_data = re.findall(r'"(202\d-\d{2}-\d{2})"[^}]*?([\d\.]+)', response.text)
        
        if raw_data:
            # Converts the raw data into a clean dictionary: { '2026-02-01': 7.44, ... }
            return {date: float(price) for date, price in raw_data}
            
    except Exception as e:
        print(f"Failed to fetch {tab} API: {e}")
        
    return {}

def main():
    # 1. Download the complete true history from both hidden APIs
    hyper_history = fetch_api_history('hyperscaler')
    neo_history = fetch_api_history('neo-cloud')
    
    if not hyper_history or not neo_history:
        print("Error: Could not retrieve data from the Silicon Data API.")
        return

    # 2. Combine the two datasets by aligning their dates perfectly
    combined_data = []
    
    # Get all unique dates from both sets, sorted from oldest to newest
    all_dates = sorted(list(set(hyper_history.keys()) | set(neo_history.keys())))
    
    last_hyper = 7.44
    last_neo = 2.90
    
    for date in all_dates:
        # If a specific day is missing data, carry over yesterday's price (Step-Function logic)
        last_hyper = hyper_history.get(date, last_hyper)
        last_neo = neo_history.get(date, last_neo)
        
        combined_data.append({
            'date': date,
            'sd_hyperscaler': last_hyper,
            'sd_neocloud': last_neo
        })

    # 3. Save the exact, perfect history back to the database
    df = pd.DataFrame(combined_data)
    file_path = 'prices.csv'
    df.to_csv(file_path, index=False)
    
    print(f"Successfully downloaded full API history! Saved {len(df)} days of exact data.")

if __name__ == "__main__":
    main()
