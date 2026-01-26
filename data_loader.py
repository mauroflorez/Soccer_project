import pandas as pd
import requests
import io
import os

# Configuration
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "matches.csv")
BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
# Seasons to fetch: 1819, 1920, 2021, 2122, 2223, 2324, 2425
SEASONS = ['1819', '1920', '2021', '2122', '2223', '2324', '2425', '2526']

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def fetch_data():
    all_data = []
    
    for season in SEASONS:
        url = BASE_URL.format(season)
        print(f"Downloading data for season {season} from {url}...")
        
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                # Read CSV from string content
                df = pd.read_csv(io.StringIO(response.text))
                df['Season'] = season  # Add season column
                all_data.append(df)
                print(f"Successfully loaded {len(df)} rows.")
            else:
                print(f"Failed to download {season}. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {season}: {e}")
            
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def save_data(df):
    if df.empty:
        print("No data to save.")
        return

    if not os.path.exists(DATA_DIR):
        print(f"Creating directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)
        
    print(f"Saving {len(df)} rows to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Data saved successfully.")

def main():
    print("Starting data retrieval...")
    df = fetch_data()
    save_data(df)
    print("Done.")

if __name__ == "__main__":
    main()
