import sqlite3
import yfinance as yf
import pandas as pd
import itertools
from statsmodels.tsa.stattools import coint, adfuller

def fetch_tickers_from_db(db_path):
    """Fetch tickers from the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM tickers")
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tickers
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

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

    # Filter pairs with cointegration p-value < 0.05 and ADF p-value < 0.05
    filtered_results = results_df[
        (results_df['Cointegration P-Value'] < 0.05) &
        (results_df['ADF P-Value'] < 0.05)
    ]

    if filtered_results.empty:
        print("No pairs meet the criteria for cointegration and stationarity.")
        return None

    # Find the best pair based on the lowest Cointegration P-Value
    best_pair = filtered_results.sort_values(by='Cointegration P-Value').iloc[0]

    # Add context to the output
    print("\nBest Cointegrated Pair and Statistical Tests:")
    print(f"Pair: {best_pair['Stock 1']} and {best_pair['Stock 2']}")
    print(f"\nCointegration P-Value: {best_pair['Cointegration P-Value']:.5f}")
    print("  - A p-value below 0.05 indicates a strong likelihood that these stocks are cointegrated, meaning they share a long-term equilibrium relationship.")
    print(f"\nADF P-Value: {best_pair['ADF P-Value']:.5f}")
    print("  - A p-value below 0.05 suggests the spread between these stocks is stationary (mean-reverting), which is ideal for pairs trading.")
    print(f"\nCorrelation: {best_pair['Correlation']:.5f}")
    print("  - Indicates the strength and direction of the linear relationship between the stocks. Values close to 1 imply a strong positive correlation.")

    return best_pair

def main():
    # Path to the SQLite database
    db_path = 'tickers.db'

    # Fetch tickers from the database
    tickers = fetch_tickers_from_db(db_path)
    if not tickers:
        print("No tickers found. Exiting.")
        return

    print(f"Tickers fetched from database: {tickers}")

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
