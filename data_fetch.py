import pandas as pd
import requests

tiingo_api_token = 'a7b701cb1a317292fe3d18abbe23e572cc297a5a'

# Function to fetch data from Tiingo API
def fetch_data(symbol, start_date, end_date):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Token {tiingo_api_token}'
    }
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {
        'startDate': start_date,
        'endDate': end_date
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        else:
            print(f"No data received for symbol {symbol}")
            return None
    else:
        print(f"Error fetching data: {response.status_code}, {response.text}")
        return None
