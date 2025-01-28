import pandas as pd
import matplotlib.pyplot as plt
from ib_insync import *
import numpy as np

# Define a pairs trading strategy
def pairs_trading_strategy(asset_x_data, asset_y_data):
    combined_data = pd.concat([asset_x_data, asset_y_data], axis=1).dropna()
    if combined_data.empty:
        print("No overlapping data. Ensure indices match and data is valid.")
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), [], []

    asset_x_data, asset_y_data = combined_data.iloc[:, 0], combined_data.iloc[:, 1]
    spread = asset_x_data - asset_y_data

    if spread.empty or spread.std() == 0 or spread.isna().all():
        print("Spread is constant or invalid. No trading opportunities.")
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), [], []

    zscore = (spread - spread.mean()) / spread.std()
    if zscore.isna().all():
        print("Z-Score calculation failed. Check data integrity.")
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), [], []

    buy_signals = zscore[zscore < -1]
    sell_signals = zscore[zscore > 1]

    print(f"Spread:\n{spread.head()}")
    print(f"Z-Scores:\n{zscore.head()}")
    print(f"Buy Signals: {len(buy_signals)}, Sell Signals: {len(sell_signals)}")
    return spread, zscore, buy_signals, sell_signals

def fetch_delayed_data(ib, ticker, duration, bar_size, shared_index=None):
    # Fetch historical price data for the last 5 years
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=5)
    bars = ib.reqHistoricalData(
        contract=ticker,
        endDateTime=end_date.strftime('%Y%m%d %H:%M:%S'),
        durationStr='5 Y',
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True
    )
    if not bars:
        print(f"No data returned for {ticker.symbol}")
        return pd.Series(dtype='float64')

    data = util.df(bars)
    data.set_index('date', inplace=True)
    return data['close']

def sharpe_ratio(pnl, risk_free_rate=0.02):
    returns = pnl.pct_change().dropna()
    if returns.empty or returns.std() == 0:
        print("No valid returns for Sharpe Ratio calculation.")
        return np.nan, np.nan
    mean_return = returns.mean() * len(returns)  # Adjust based on actual trading days
    std_dev = returns.std() * np.sqrt(len(returns))  # Scale by actual period
    sharpe = (mean_return - risk_free_rate) / std_dev
    return sharpe, mean_return * 100

def backtest_pairs_trading(asset_x_data, asset_y_data, entry_threshold=0.5, exit_threshold=0.2, transaction_cost=0.005):
    spread, zscore, buy_signals, sell_signals = pairs_trading_strategy(asset_x_data, asset_y_data)
    if spread.empty or zscore.empty:
        print("Spread or Z-Score is empty. Exiting backtest.")
        return pd.DataFrame(), 0

    position = None
    pnl = []
    cash = 100000  # Starting cash balance
    dates = []
    entry_price = None

    for date, current_zscore in zscore.dropna().items():
        position_size = min(10000, cash * 0.1)  # Dynamic position sizing based on cash balance
        if current_zscore < -entry_threshold and position is None:
            position = 'long'
            entry_price = spread.loc[date]
            cash -= position_size * transaction_cost
        elif current_zscore > entry_threshold and position is None:
            position = 'short'
            entry_price = spread.loc[date]
            cash -= position_size * transaction_cost
        elif abs(current_zscore) < exit_threshold and position is not None:
            if position == 'long':
                pnl_value = position_size * (spread.loc[date] - entry_price) - position_size * transaction_cost
                pnl.append(pnl_value)
            elif position == 'short':
                pnl_value = position_size * (entry_price - spread.loc[date]) - position_size * transaction_cost
                pnl.append(pnl_value)
            cash += pnl[-1]
            position = None
            entry_price = None
        else:
            pnl.append(0 if len(pnl) == 0 else pnl[-1])

        dates.append(date)

    # Ensure dates and pnl are the same length
    min_length = min(len(dates), len(pnl))
    dates = dates[:min_length]
    pnl = pnl[:min_length]

    # Validate PnL for NaN or infinite values
    pnl = np.array(pnl)
    if np.isnan(pnl).any() or np.isinf(pnl).any():
        print("PnL contains invalid values. Exiting backtest.")
        return pd.DataFrame(), cash

    pnl_df = pd.DataFrame({'Date': dates, 'PnL': np.cumsum(pnl)})
    pnl_df.set_index('Date', inplace=True)
    return pnl_df, cash

def main():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
    except Exception as e:
        print(f"Could not connect to Interactive Brokers: {e}")
        return

    shared_index = pd.date_range(end=pd.Timestamp.today(), periods=1826)  # 5 years of data
    asset_x_data = fetch_delayed_data(ib, Stock('MDT', 'SMART', 'USD'), "5 Y", "1 day", shared_index)
    asset_y_data = fetch_delayed_data(ib, Stock('YUM', 'SMART', 'USD'), "5 Y", "1 day", shared_index)

    if asset_x_data.empty or asset_y_data.empty:
        print("Failed to fetch data. Exiting.")
        return

    results, cash = backtest_pairs_trading(asset_x_data, asset_y_data)
    if results.empty:
        print("No valid results from backtesting. Exiting.")
        return

    cumulative_pnl = results['PnL']
    if cumulative_pnl.empty:
        print("Cumulative PnL is empty. No trades executed.")
        return

    cumulative_pnl.plot(title='Cumulative PnL', figsize=(12, 6))
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.grid(True)
    plt.show()

    sharpe, mean_percentage_return = sharpe_ratio(cumulative_pnl)
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Mean Return (%): {mean_percentage_return:.2f}%")

if __name__ == "__main__":
    main()
