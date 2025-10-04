# technical_indicators.py
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    在K線DataFrame中新增技術指標
    必須包含 ['Open', 'High', 'Low', 'Close', 'Volume'] 欄位
    """

    # === 移動平均線 (MA) ===
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()

    # === RSI (14) ===
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    RS = roll_up / (roll_down + 1e-10)  # 避免除以零
    df['RSI14'] = 100.0 - (100.0 / (1.0 + RS))

    # === MACD (12, 26, 9) ===
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df
