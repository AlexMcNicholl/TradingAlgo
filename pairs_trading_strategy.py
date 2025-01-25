import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ib_insync import *

def fetch_delayed_data(ib, ticker, duration, bar_size):
    """Fetch delayed historical data from IB API."""
    bars = ib.reqHistoricalData(
        contract=ticker,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    if not bars:
        print(f"No data returned for {ticker.symbol}")
        return pd.DataFrame()
    
    data = util.df(bars)
    if 'date' in data.columns:
        data.set_index('date', inplace=True)
    return data['close']

def calculate_zscore(spread, window=30):
    """Calculate the z-score for the spread."""
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore

def plot_zscore_with_signals(zscore, buy_signals, sell_signals):
    """Plot the Z-Score with Buy/Sell signals."""
    plt.figure(figsize=(12, 6))

    # Plot the Z-Score
    plt.plot(zscore.index, zscore, label='Z-Score', color='blue')

    # Plot Buy and Sell signals
    plt.scatter(buy_signals.index, buy_signals, color='green', marker='^', label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals, color='red', marker='v', label='Sell Signal')

    # Plot thresholds
    plt.axhline(0, color='gray', linestyle='--', label='Mean')
    plt.axhline(1.0, color='purple', linestyle='--', label='Sell Threshold (+1.0)')
    plt.axhline(-1.0, color='purple', linestyle='--', label='Buy Threshold (-1.0)')

    plt.title('Z-Score with Buy/Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Z-Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Connect to Interactive Brokers
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    # Define stocks
    ticker1 = Stock('LUMN', 'SMART', 'USD')
    ticker2 = Stock('VOD', 'SMART', 'USD')

    # Fetch delayed data
    duration = "120 d"
    bar_size = "1 day"
    duk_data = fetch_delayed_data(ib, ticker1, duration, bar_size)
    so_data = fetch_delayed_data(ib, ticker2, duration, bar_size)

    if duk_data.empty or so_data.empty:
        print("Failed to fetch data. Exiting.")
        return

    # Calculate the spread and z-score
    spread = duk_data - so_data
    zscore = calculate_zscore(spread, window=30)

    # Identify buy and sell signals
    buy_signals = zscore[zscore < -1.0]
    sell_signals = zscore[zscore > 1.0]

    # Plot the Z-Score with buy/sell signals
    plot_zscore_with_signals(zscore, buy_signals, sell_signals)

if __name__ == "__main__":
    main()
