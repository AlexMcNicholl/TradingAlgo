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

def fetch_tickers_from_api():
    """Fetch tickers dynamically using IBKR API for each asset class."""
    equities = [Stock(ticker, 'SMART', 'USD') for ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']]
    commodities = [Future(symbol, exchange) for symbol, exchange in [('CL', 'NYMEX'), ('GC', 'COMEX'), ('SI', 'COMEX')]]
    forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDJPY=X']  # Update to Yahoo Finance-compatible symbols

    # Qualify contracts via IBKR (for equities and commodities)
    qualified_equities = ib.qualifyContracts(*equities)
    qualified_commodities = ib.qualifyContracts(*commodities)

    # Combine tickers into a single list
    return [contract.symbol for contract in qualified_equities + qualified_commodities] + forex_pairs

def fetch_data(tickers):
    """Fetch historical data for given tickers using yfinance."""
    try:
        data = yf.download(tickers, start='2023-01-01', end='2024-01-01')['Adj Close']
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def test_cointegration(series1, series2):
    """Perform cointegration test and return p-value."""
    score, p_value, _ = coint(series1, series2)
    return p_value

def test_adf(spread):
    """Perform ADF test on the spread and return p-value."""
    result = adfuller(spread)
    return result[1]

def calculate_dynamic_zscore(spread, window=30):
    """Calculate a rolling Z-score for the spread."""
    rolling_mean = spread.rolling(window=min(len(spread), window)).mean()
    rolling_std = spread.rolling(window=min(len(spread), window)).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore.interpolate().rolling(window=5).mean().dropna()


def rolling_cointegration(y, x, window=120):
    """Perform rolling cointegration tests and return p-values."""
    results = []
    for start in range(len(y) - window):
        end = start + window
        p_value = test_cointegration(y[start:end], x[start:end])
        results.append(p_value)
    return pd.Series(results, index=y.index[window:])

def hurst_exponent(series, max_lag=100):
    """Calculate the Hurst exponent to detect mean-reversion (H < 0.5)."""
    lags = range(2, max_lag)
    tau = [np.std(series.diff(lag)) for lag in lags]
    hurst = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return hurst

def adaptive_thresholds(spread, window=60):
    """Calculate adaptive buy/sell thresholds based on spread volatility."""
    rolling_std = spread.rolling(window).std()
    upper_threshold = rolling_std * 2
    lower_threshold = -rolling_std * 2
    return upper_threshold, lower_threshold

def plot_spread(series1, series2, spread, zscore, p_value, adf_value, correlation):
    """Plot the spread of the identified pair and display test results."""
    plt.figure(figsize=(14, 8))

    # Plot asset prices (primary y-axis)
    ax1 = plt.subplot(211)
    ax1.plot(series1.index, series1, label='Asset 1', color='blue')
    ax1.plot(series2.index, series2, label='Asset 2', color='orange')
    ax1.set_title('Price Movement of Assets', fontsize=14)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot spread and Z-score (secondary y-axis)
    ax2 = plt.subplot(212)
    ax2.plot(spread.index, spread, label='Spread', color='green')
    ax2.axhline(spread.mean(), color='red', linestyle='--', label='Spread Mean')
    ax2.set_ylabel('Spread', fontsize=12)
    
    ax3 = ax2.twinx()
    ax3.plot(zscore.index, zscore, label='Z-Score', color='purple', linestyle='dotted')
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_ylabel('Z-Score', fontsize=12)

    # Display statistical results on the graph
    textstr = (
        f"Cointegration P-Value: {p_value:.5f}\\n"
        f"ADF P-Value: {adf_value:.5f}\\n"
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


def identify_cointegrated_pairs(data):
    """Identify cointegrated pairs and return the pair with the best statistical values."""
    pairs = list(itertools.combinations(data.columns, 2))
    results = []

    for pair in pairs:
        stock1, stock2 = pair
        series1 = data[stock1].dropna()
        series2 = data[stock2].dropna()

        if series1.empty or series2.empty:
            continue

        # Align series on the same dates
        combined = pd.concat([series1, series2], axis=1).dropna()
        if combined.empty:
            continue

        series1, series2 = combined[stock1], combined[stock2]
        spread = series1 - series2

        # Perform statistical tests
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

    # Create a DataFrame of results
    results_df = pd.DataFrame(results)

    # Show all tested pairs for debugging
    print("\nAll Pair Test Results:")
    print(results_df)

    # Filter pairs with cointegration p-value < 0.05 and ADF p-value < 0.05
    filtered_results = results_df[
        (results_df['Cointegration P-Value'] < 0.05) &
        (results_df['ADF P-Value'] < 0.05)
    ]

    if filtered_results.empty:
        print("No pairs meet the criteria for cointegration and stationarity.")
        # Fallback: Use the pair with the lowest Cointegration P-Value
        best_pair = results_df.sort_values(by='Cointegration P-Value').iloc[0]
        print("\nFallback Pair (Lowest Cointegration P-Value):")
        print(f"Pair: {best_pair['Stock 1']} and {best_pair['Stock 2']}")
        print(f"Cointegration P-Value: {best_pair['Cointegration P-Value']:.5f}")
        print(f"ADF P-Value: {best_pair['ADF P-Value']:.5f}")
        print(f"Correlation: {best_pair['Correlation']:.5f}")
    else:
        best_pair = filtered_results.sort_values(by='Cointegration P-Value').iloc[0]

    # Calculate spread and Z-score for the best or fallback pair
    spread = data[best_pair['Stock 1']] - data[best_pair['Stock 2']]
    zscore = calculate_dynamic_zscore(spread)

    # Validate fallback pair with Hurst exponent
    hurst = hurst_exponent(spread)
    print(f"Hurst Exponent: {hurst:.5f}")
    if hurst >= 0.5:
        print("Warning: Spread may not be mean-reverting.")

    # Plot the spread and Z-score of the best or fallback pair
    plot_spread(data[best_pair['Stock 1']], data[best_pair['Stock 2']], spread, zscore,
                best_pair['Cointegration P-Value'], best_pair['ADF P-Value'], best_pair['Correlation'])

    return best_pair

def main():
    # Fetch tickers dynamically
    tickers = fetch_tickers_from_api()
    if not tickers:
        print("No tickers found. Exiting.")
        return

    print(f"Tickers fetched: {tickers}")

    # Fetch historical data
    print("Fetching data...")
    data = fetch_data(tickers)
    if data.empty:
        print("Failed to fetch data. Exiting.")
        return

    print("Data fetched successfully!")
    print("Identifying cointegrated pairs...")

    # Identify the best cointegrated pair
    identify_cointegrated_pairs(data)

if __name__ == "__main__":
    main()
