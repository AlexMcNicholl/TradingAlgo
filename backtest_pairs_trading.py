import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG, EURUSD
from ib_insync import *

# Custom function to compute ATR using NumPy

def ATR(df, n=20):
    high_low = df.High - df.Low
    high_close = np.abs(df.High - np.roll(df.Close, shift=1))
    low_close = np.abs(df.Low - np.roll(df.Close, shift=1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return pd.Series(true_range).rolling(n).mean().to_numpy()

class PairsTradingStrategy(Strategy):
    entry_threshold = 2.0  # Increased for better filtering
    exit_threshold = 1.0  # Increased to reduce false exits
    transaction_cost = 0.005
    atr_multiplier = 2  # ATR-based stop-loss multiplier
    
    def init(self):
        price_x = self.data.Close
        price_y = self.data.y
        spread = price_x - price_y
        
        mean_spread = self.I(lambda x: pd.Series(x).rolling(window=20).mean(), spread)
        std_spread = self.I(lambda x: pd.Series(x).rolling(window=20).std(), spread)
        self.zscore = self.I(lambda x: (pd.Series(x) - mean_spread) / std_spread, spread)
        self.atr = self.I(ATR, self.data, 20)  # Using custom ATR function

    def next(self):
        atr_value = self.atr[-1] if self.atr[-1] > 0 else 0.01  # Ensure ATR is never zero
        stop_loss_price = self.data.Close[-1] - (self.atr_multiplier * atr_value) if self.position.is_long else self.data.Close[-1] + (self.atr_multiplier * atr_value)
        
        # Ensure SL placement follows Backtesting.py's rules
        if self.zscore[-1] < -self.entry_threshold:
            stop_loss_price = min(stop_loss_price, self.data.Close[-1] * 0.98)  # Ensure SL is below entry for long trades
            self.buy(sl=stop_loss_price)
        elif self.zscore[-1] > self.entry_threshold:
            stop_loss_price = max(stop_loss_price, self.data.Close[-1] * 1.02)  # Ensure SL is above entry for short trades
            self.sell(sl=stop_loss_price)
        elif abs(self.zscore[-1]) < self.exit_threshold:
            self.position.close()

def fetch_delayed_data(ib, ticker, duration, bar_size):
    bars = ib.reqHistoricalData(
        contract=ticker,
        endDateTime=pd.Timestamp.today().strftime('%Y%m%d %H:%M:%S'),
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True
    )
    if not bars:
        print(f"No data returned for {ticker.symbol}")
        return pd.DataFrame()
    data = util.df(bars)
    data.set_index('date', inplace=True)
    data.index = pd.to_datetime(data.index)  # Ensure DateTime index
    return data[['open', 'high', 'low', 'close', 'volume']]

def main():
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)
    except Exception as e:
        print(f"Could not connect to Interactive Brokers: {e}")
        return

    asset_x_data = fetch_delayed_data(ib, Stock('PFE', 'SMART', 'USD'), "5 Y", "1 day")
    asset_y_data = fetch_delayed_data(ib, Stock('FXI', 'SMART', 'USD'), "5 Y", "1 day")
    
    if asset_x_data.empty or asset_y_data.empty:
        print("Failed to fetch data. Exiting.")
        return
    
    df = asset_x_data.copy()
    df['y'] = asset_y_data['close']
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)  # Ensure DateTime index
    
    backtest = Backtest(df, PairsTradingStrategy, cash=100000, commission=0.001)
    results = backtest.run()
    backtest.plot()
    
    print(results)

if __name__ == "__main__":
    main()
