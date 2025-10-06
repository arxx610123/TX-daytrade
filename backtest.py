# backtest.py
import pandas as pd
import numpy as np

def calculate_indicators(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def generate_signals(df):
    df['Signal_Flag'] = "HOLD"
    for i in range(1, len(df)):
        if df['MA5'].iloc[i] > df['MA20'].iloc[i] and df['MA5'].iloc[i-1] <= df['MA20'].iloc[i-1]:
            df.loc[df.index[i], 'Signal_Flag'] = "BUY"
        elif df['MA5'].iloc[i] < df['MA20'].iloc[i] and df['MA5'].iloc[i-1] >= df['MA20'].iloc[i-1]:
            df.loc[df.index[i], 'Signal_Flag'] = "SELL"
    return df

def backtest(df, initial_cash=1000000):
    cash = initial_cash
    position = 0
    trade_log = []
    entry_price = 0

    for i, row in df.iterrows():
        price = row['Close']
        signal = row['Signal_Flag']

        if signal == "BUY" and position == 0:
            position = 1
            entry_price = price
            trade_log.append((row['datetime'], "BUY", price, cash))
        elif signal == "SELL" and position == 1:
            profit = price - entry_price
            cash += profit
            position = 0
            trade_log.append((row['datetime'], "SELL", price, cash))

    total_return = cash - initial_cash
    return total_return, trade_log

if __name__ == "__main__":
    file_path = "data/5m_day/txf_5m_day_20251002.csv"
    df = pd.read_csv(file_path)

    # 修正這裡：讀取小寫 datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise KeyError("CSV 檔裡找不到 'datetime' 欄位")

    df = calculate_indicators(df)
    df = generate_signals(df)
    total_return, trade_log = backtest(df)

    print("=== 回測結果 ===")
    print(f"總報酬: {total_return:.2f}")
    print(f"交易次數: {len([t for t in trade_log if t[1]=='BUY'])}")
    print("\n=== 交易紀錄 ===")
    for t in trade_log:
        print(t)
