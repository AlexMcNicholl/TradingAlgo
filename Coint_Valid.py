import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
from ib_insync import IB, Stock, Future, Forex

# Initialize IBKR API connection
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # Replace 7497 with 4002 if using IB Gateway

def fetch_sp500_tickers():
    """Fetch all S&P 500 tickers dynamically from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url, header=0)[0]
    return sp500_table['Symbol'].tolist()

def fetch_commodity_tickers_from_ib():
    """Fetch available commodity futures tickers dynamically from IBKR."""
    commodity_symbols = ['CL', 'GC', 'SI', 'ZC', 'NG']  # Add more as needed
    futures = [Future(symbol, 'NYMEX') for symbol in commodity_symbols]
    qualified_contracts = ib.qualifyContracts(*futures)
    return [contract.symbol for contract in qualified_contracts]

def fetch_forex_tickers_from_ib():
    """Fetch forex pairs dynamically from IBKR."""
    forex_pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'EURGBP', 'AUDJPY']
    forex = [Forex(pair) for pair in forex_pairs]
    qualified_forex = ib.qualifyContracts(*forex)
    return [contract.symbol for contract in qualified_forex]

def fetch_all_tickers():
    """Fetch all asset tickers dynamically."""
    equities = fetch_sp500_tickers()  # Large equity universe
    commodities = fetch_commodity_tickers_from_ib()
    forex = fetch_forex_tickers_from_ib()
    return equities + commodities + forex

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
    score, p_value, _ = coint(series1, series2)
    return p_value

def test_adf(spread):
    """Perform ADF test on the spread and return p-value."""
    result = adfuller(spread)
    return result[1]

def calculate_dynamic_zscore(spread, window=60):
    """Calculate an exponentially weighted rolling Z-score for the spread."""
    rolling_mean = spread.ewm(span=window).mean()
    rolling_std = spread.ewm(span=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore.dropna()

def plot_spread(series1, series2, spread, zscore, p_value, adf_value, correlation):
    """Plot the spread of the identified pair and display test results."""
    plt.figure(figsize=(14, 8))

    # Plot asset prices
    ax1 = plt.subplot(211)
    ax1.plot(series1.index, series1, label='Asset 1', color='blue')
    ax1.plot(series2.index, series2, label='Asset 2', color='orange')
    ax1.set_title('Price Movement of Assets', fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot spread and Z-score
    ax2 = plt.subplot(212)
    ax2.plot(spread.index, spread, label='Spread', color='green')
    ax2.axhline(spread.mean(), color='red', linestyle='--', label='Spread Mean')
    ax2.set_ylabel('Spread', fontsize=12)
    
    ax3 = ax2.twinx()
    ax3.plot(zscore.index, zscore, label='Z-Score', color='purple', linestyle='dotted')
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_ylabel('Z-Score', fontsize=12)

    # Display statistical results
    textstr = (
        f"Cointegration P-Value: {p_value:.5f}\n"
        f"ADF P-Value: {adf_value:.5f}\n"
        f"Correlation: {correlation:.5f}"
    )
    ax2.text(0.75, 0.05, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray'))

    ax2.set_title('Spread and Z-Score of Identified Pair', fontsize=14)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def test_pairs(data, pairs):
    """Test all filtered pairs for cointegration and stationarity."""
    results = []
    for stock1, stock2 in pairs:
        series1 = data[stock1].dropna()
        series2 = data[stock2].dropna()
        if series1.empty or series2.empty:
            continue

        # Perform cointegration and ADF tests
        spread = series1 - series2
        coint_p_value = test_cointegration(series1, series2)
        adf_p_value = test_adf(spread)
        correlation = series1.corr(series2)

        results.append({
            'Stock 1': stock1,
            'Stock 2': stock2,
            'Cointegration P-Value': coint_p_value,
            'ADF P-Value': adf_p_value,
            'Correlation': correlation
        })
    return pd.DataFrame(results)

def identify_cointegrated_pairs(data):
    """Identify cointegrated pairs and return the pair with the best statistical values."""
    # Pre-filter pairs with high correlation
    print("Filtering high-correlation pairs...")
    pairs = filter_high_correlation(data)

    # Test cointegration and stationarity
    print(f"Testing {len(pairs)} pairs for cointegration...")
    results = test_pairs(data, pairs)

    # Display results
    print("\nAll Pair Test Results:")
    print(results)

    # Filter pairs meeting criteria
    best_pairs = results[
        (results['Cointegration P-Value'] < 0.05) & (results['ADF P-Value'] < 0.05)
    ]
    if best_pairs.empty:
        print("No pairs meet the criteria.")
        best_pair = results.sort_values(by='Cointegration P-Value').iloc[0]
    else:
        print("Best Cointegrated Pairs:")
        print(best_pairs)
        best_pair = best_pairs.sort_values(by='Cointegration P-Value').iloc[0]

    # Calculate spread and Z-score for the best pair
    spread = data[best_pair['Stock 1']] - data[best_pair['Stock 2']]
    zscore = calculate_dynamic_zscore(spread)

    plot_spread(data[best_pair['Stock 1']], data[best_pair['Stock 2']], spread, zscore,
                best_pair['Cointegration P-Value'], best_pair['ADF P-Value'], best_pair['Correlation'])

def main():
    # Fetch all asset tickers dynamically
    tickers = fetch_all_tickers()
    print(f"Fetched {len(tickers)} tickers.")

    # Fetch historical data
    print("Fetching historical data...")
    data = fetch_data(tickers)
    if data.empty:
        print("Failed to fetch data. Exiting.")
        return

    # Filter sparse data
    print("Filtering sparse data...")
    data = filter_sparse_data(data)

    # Identify the best cointegrated pair
    identify_cointegrated_pairs(data)

if __name__ == "__main__":
    main()
