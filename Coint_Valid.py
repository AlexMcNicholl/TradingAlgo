import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
from ib_insync import IB, Stock, ScannerSubscription
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize IBKR API connection
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Replace 7497 with 4002 if using IB Gateway

def fetch_energy_tickers_from_ib(limit=None):
    """Fetch available energy sector stocks dynamically from IBKR."""
    scan_sub = ScannerSubscription(
        numberOfRows=limit or 50,
        instrument='STK',
        locationCode='STK.US.MAJOR',
        scanCode='MOST_ACTIVE'
    )
    energy_stocks = ib.reqScannerData(scan_sub)
    
    # Extract symbols correctly
    tickers = [stock.contractDetails.contract.symbol for stock in energy_stocks]
    
    if not tickers:
        print("No energy tickers found.")
        return []
    
    return tickers[:limit] if limit else tickers

def fetch_data(tickers):
    """Fetch historical data for given tickers using yfinance."""
    try:
        data = yf.download(tickers, start='2023-01-01', end='2024-01-01')['Adj Close']
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def filter_sparse_data(data):
    """Filter out assets with sparse or missing data."""
    return data.dropna(thresh=int(0.8 * len(data.index)), axis=1)  # Keep tickers with at least 80% valid data

def filter_high_correlation(data, threshold=0.8):
    """Filter pairs based on high correlation."""
    corr_matrix = data.corr()
    pairs = [(i, j) for i in corr_matrix.columns for j in corr_matrix.columns if i != j and abs(corr_matrix.loc[i, j]) > threshold]
    return pairs

def test_cointegration(series1, series2):
    """Perform cointegration test and return p-value."""
    # Align series to have the same time index
    series1, series2 = series1.align(series2, join='inner')
    
    # Ensure non-empty, equal-sized series before testing
    if len(series1) == 0 or len(series2) == 0:
        return np.nan  # Return NaN if the pair is invalid
    
    score, p_value, _ = coint(series1, series2)
    return p_value

def test_adf(spread):
    """Perform ADF test on the spread and return p-value."""
    result = adfuller(spread.dropna())
    return result[1]

def calculate_dynamic_zscore(spread, window=60):
    """Calculate an exponentially weighted rolling Z-score for the spread."""
    rolling_mean = spread.ewm(span=window).mean()
    rolling_std = spread.ewm(span=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore.dropna()

def test_pairs(data, pairs):
    """Test all filtered pairs for cointegration and stationarity."""
    results = []
    for stock1, stock2 in pairs:
        series1 = data[stock1].dropna()
        series2 = data[stock2].dropna()
        
        # Align time series before testing
        series1, series2 = series1.align(series2, join='inner')
        
        if len(series1) < 30 or len(series2) < 30:  # Ensure sufficient data
            continue

        # Perform cointegration and ADF tests
        spread = series1 - series2
        coint_p_value = test_cointegration(series1, series2)
        adf_p_value = test_adf(spread)
        correlation = series1.corr(series2)
        
        if not np.isnan(coint_p_value):  # Only append valid results
            results.append({
                'Stock 1': stock1,
                'Stock 2': stock2,
                'Cointegration P-Value': coint_p_value,
                'ADF P-Value': adf_p_value,
                'Correlation': correlation
            })
    
    return pd.DataFrame(results)

def identify_cointegrated_pairs(data):
    """Identify cointegrated pairs and return the top-ranked statistical values."""
    print("Filtering high-correlation pairs...")
    pairs = filter_high_correlation(data)
    
    print(f"Testing {len(pairs)} pairs for cointegration...")
    results = test_pairs(data, pairs)
    
    print("\nAll Pair Test Results:")
    print(results)
    
    best_pairs = results[
        (results['Cointegration P-Value'] < 0.05) & (results['ADF P-Value'] < 0.05)
    ]
    
    if best_pairs.empty:
        print("No pairs meet the criteria.")
        return None
    
    best_pairs_sorted = best_pairs.sort_values(by='Cointegration P-Value').head(3)
    
    print("\nTop 3 Cointegrated Pairs:")
    print(best_pairs_sorted)
    
    return best_pairs_sorted

def main():
    tickers = fetch_energy_tickers_from_ib(limit=50)  # Fetch large list of energy stocks
    print(f"Fetched {len(tickers)} tickers.")
    
    print("Fetching historical data...")
    data = fetch_data(tickers)
    if data.empty:
        print("Failed to fetch data. Exiting.")
        return
    
    print("Filtering sparse data...")
    data = filter_sparse_data(data)
    
    identify_cointegrated_pairs(data)

if __name__ == "__main__":
    main()
