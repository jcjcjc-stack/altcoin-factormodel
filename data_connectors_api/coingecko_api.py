import requests
import pandas as pd

def get_market_caps(vs_currency="usd", per_page=200):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data)
    
    return df[["name", "symbol","market_cap"]]

if __name__ == "__main__":
    df = get_market_caps()
    print(df.tail())
