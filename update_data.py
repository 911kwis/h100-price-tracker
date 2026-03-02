import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

def scrape_silicon_data_hyperscaler():
    """Pulls the Tier-1 Hyperscaler H100 price directly from Silicon Data's HTML."""
    url = "https://www.silicondata.com/products/silicon-index"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        # Finds the price next to the 'USD' text on their site
        prices = re.findall(r'([0-9]+\.[0-9]{2})\s*USD', BeautifulSoup(response.text, 'html.parser').get_text())
        if prices:
            return float(prices[0])
    except Exception as e:
        print(f"Silicon Data Scrape Error: {e}")
    return 7.44

def scrape_silicon_data_neocloud():
    """
    Placeholder for the Silicon Data Neo-Cloud API.
    Because this is hidden behind a JS toggle on their site, we hold the 
    last known value until the JSON API endpoint is plugged in here.
    """
    # TODO: Replace with the direct JSON API URL from Silicon Data's network tab
    return 2.90 

def main():
    hyper_price = scrape_silicon_data_hyperscaler()
    neo_price = scrape_silicon_data_neocloud()
    
    today = datetime.now().strftime('%Y-%m-%d')
    file_path = 'prices.csv'
    
    df = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame(columns=['date', 'sd_hyperscaler', 'sd_neocloud'])
    
    new_row = pd.DataFrame({'date': [today], 'sd_hyperscaler': [hyper_price], 'sd_neocloud': [neo_price]})
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.drop_duplicates(subset=['date'], keep='last')
    df.to_csv(file_path, index=False)
    
    print(f"Silicon Data Logged -> {today} | Hyperscaler: ${hyper_price} | Neo-Cloud: ${neo_price}")

if __name__ == "__main__":
    main()
