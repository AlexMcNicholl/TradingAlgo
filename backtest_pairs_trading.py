import pandas as pd
import matplotlib.pyplot as plt
from ignore_backtestvalidity import pairs_trading_strategy, fetch_delayed_data
from ib_insync import *
import numpy as np

# DEBUGGING CODE - CAN IGNORE
import sys
import os

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Print the system path
print("System Path:", sys.path)

# Attempt to import the module
try:
    from ignore_backtestvalidity import pairs_trading_strategy, fetch_delayed_data
    print("Successfully imported pairs_trading_strategy.")
except ImportError as e:
    print("ImportError:", e)


def backtest_pairs_trading(duk_data, so_data, entry_threshold=1.0, exit_threshold=0.0):
    spread, zscore, buy_signals, sell_signals = pairs_trading_strategy(duk_data, so_data)

    position = None
    pnl = []
    dates = []

    for date, current_zscore in zscore.dropna().items():
        if current_zscore < -entry_threshold and position is None:
            position = 'long'
        elif current_zscore > entry_threshold and position is None:
            position = 'short'
        elif abs(current_zscore) < exit_threshold and position is not None:
            position = None

        if position:
            pnl.append(spread.loc[date])
            dates.append(date)

    pnl_df = pd.DataFrame({'Date': dates, 'PnL': pnl})
    pnl_df.set_index('Date', inplace=True)
    return pnl_df


# Calculate Sharpe Ratio
def sharpe_ratio(pnl, risk_free_rate=0):
    # Calculate percentage returns
    returns = pnl.pct_change().dropna()
    
    # Convert returns to percentage
    returns_percentage = returns * 100
    
    # Calculate mean of returns in percentage terms
    mean_percentage_return = returns_percentage.mean()
    
    # Handle zero standard deviation to avoid division by zero
    if returns.std() == 0:
        return 0, mean_percentage_return
    
    # Calculate Sharpe Ratio
    sharpe = (returns.mean() - risk_free_rate) / returns.std()
    
    return sharpe, mean_percentage_return


# Calculate Maximum Drawdown
def max_drawdown(pnl):
    cumulative_max = pnl.cummax()
    drawdown = pnl - cumulative_max
    return drawdown.min()

def main():
    # Connect to Interactive Brokers
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    # Fetch data for stock 1 and stock 2
    duration = "60 D"
    bar_size = "1 day"
    duk_data = fetch_delayed_data(ib, Stock('VOD', 'SMART', 'USD'), duration, bar_size)
    so_data = fetch_delayed_data(ib, Stock('LUMN', 'SMART', 'USD'), duration, bar_size)

    if duk_data.empty or so_data.empty:
        print("Failed to fetch data. Exiting.")
        return

    # Perform backtesting
    results = backtest_pairs_trading(duk_data, so_data)
    print(results)

    # Calculate cumulative PnL
    cumulative_pnl = results['PnL'].cumsum()

    # Plot cumulative PnL
    cumulative_pnl.plot(title='Cumulative PnL')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)
    plt.show()

    # Calculate and display metrics
    sharpe, mean_percentage_return = sharpe_ratio(cumulative_pnl)

    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Mean Return (%): {mean_percentage_return:.2f}%")

if __name__ == "__main__":
    main()
