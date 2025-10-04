import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from technical_indicators import add_indicators

def load_and_clean_data(file_path):
    print(">>> 開始讀取資料")

    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案: {file_path}")
        sys.exit(1)

    # 嘗試多種日期欄位名稱
    date_cols = ["Datetime", "datetime", "date", "time"]
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ 讀檔失敗: {e}")
        sys.exit(1)

    # 找到正確的日期欄位並轉換
    found_date = None
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                df.rename(columns={col: "Datetime"}, inplace=True)
                found_date = "Datetime"
                break
            except Exception:
                continue
    if not found_date:
        print("⚠️ 找不到日期欄位，將使用索引代替")
        df["Datetime"] = pd.date_range(start="2000-01-01", periods=len(df), freq="T")

    # 欄位統一名稱
    col_map = {c.lower(): c for c in df.columns}
    rename_dict = {}
    for std in ["open", "high", "low", "close", "volume"]:
        if std in col_map:
            rename_dict[col_map[std]] = std.capitalize()
    df.rename(columns=rename_dict, inplace=True)

    # 確保必要欄位存在
    required_cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 缺少必要欄位: {col}")
            sys.exit(1)

    print(f">>> 成功讀取 {len(df)} 筆資料，欄位: {list(df.columns)}")

    # 去除缺失值
    df.dropna(inplace=True)

    return df


def parameter_tuning(df):
    print(">>> 開始進行 GridSearchCV 調參")

    # 加入技術指標
    df = add_indicators(df)

    # 目標 (漲跌 >0 = 1, else 0)
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Target"] = (df["Return"] > 0).astype(int)

    features = df[["MA5", "MA10", "RSI14", "MACD", "Signal"]]
    target = df["Target"]

    if len(df) < 10:
        print("❌ 資料不足，無法調參")
        sys.exit(1)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(random_state=42, max_iter=1000, tol=1e-3))
    ])

    param_grid = {
        "clf__loss": ["hinge", "log_loss"],
        "clf__penalty": ["l2", "l1"],
        "clf__alpha": [0.0001, 0.001, 0.01]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(features, target)

    print(">>> 調參完成 ✅")
    print("最佳參數:", grid.best_params_)
    print("最佳準確率:", round(grid.best_score_, 4))

    return grid.best_estimator_


def backtest(df, model):
    print(">>> 開始模擬交易回測")

    df = add_indicators(df)
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["Target"] = (df["Return"] > 0).astype(int)

    features = df[["MA5", "MA10", "RSI14", "MACD", "Signal"]]
    preds = model.predict(features)

    balance = 1_000_000
    position = None
    trade_log = []

    for i in range(len(df)):
        price = df["Close"].iloc[i]
        ts = df["Datetime"].iloc[i]
        signal = preds[i]

        if signal == 1 and position is None:
            position = price
            trade_log.append((ts, "BUY", float(price)))
        elif signal == 0 and position is not None:
            profit = price - position
            balance += profit
            trade_log.append((ts, "SELL", float(price)))
            position = None

    print(">>> 回測完成 ✅")
    print(f"總報酬: {balance:.2f}")
    print(f"交易次數: {len(trade_log)}")
    print("=== 交易紀錄 ===")
    for log in trade_log:
        print(log)


if __name__ == "__main__":
    print(">>> backtest_tune.py 啟動")

    if len(sys.argv) < 2:
        print("❌ 請提供 CSV 檔案路徑")
        sys.exit(1)

    file_path = sys.argv[1]
    df = load_and_clean_data(file_path)
    model = parameter_tuning(df)
    backtest(df, model)

    print(">>> backtest_tune.py 結束")
