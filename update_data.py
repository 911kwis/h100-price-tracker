import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os

def scrape_live_h100_price():
    url = "https://gpus.io/gpus/h100"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        prices = re.findall(r'\$\s*([0-9]+\.[0-9]{2})', soup.get_text())
        valid_prices = [float(p) for p in prices if 1.0 <= float(p) <= 10.0]
        if valid_prices:
            return np.median(valid_prices)
    except Exception as e:
        print(f"Error scraping: {e}")
    return None

def main():
    price = scrape_live_h100_price()
    if price:
        today = datetime.now().strftime('%Y-%m-%d')
        file_path = 'prices.csv'
        
        # Load the existing database
        df = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame(columns=['date', 'price'])
        
        # Add today's data
        new_row = pd.DataFrame({'date': [today], 'price': [price]})
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Prevent duplicate entries if it runs twice in one day
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Save it back to the file
        df.to_csv(file_path, index=False)
        print(f"Successfully recorded {today}: ${price}/hr")

if __name__ == "__main__":
    main()
