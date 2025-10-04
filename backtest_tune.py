import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from technical_indicators import add_technical_indicators

# === 資料讀取 ===
def load_data(file_path):
    print(">>> 開始讀取資料")
    df = pd.read_csv(file_path)
    print(f">>> 成功讀取 {len(df)} 筆資料，欄位: {list(df.columns)}")
    return df

# === 加入技術指標 & 標註標籤 ===
def prepare_features(df):
    df = add_technical_indicators(df)
    # 移除含 NaN 的 target
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 建立漲跌標籤
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # 去除 NaN（僅限必要的）
    df = df.dropna(subset=['Target'])
    return df

# === GridSearch 調參 ===
def parameter_tuning(df):
    print(">>> 開始進行 GridSearchCV 調參")

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA20', 'RSI', 'MACD', 'Signal']
    X = df[feature_cols]
    y = df['Target']

    # Pipeline 包含 NaN 處理 + 標準化 + SGD 模型
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('clf', SGDClassifier(random_state=42))
    ])

    param_grid = {
        'clf__loss': ['log_loss', 'hinge'],
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__penalty': ['l2', 'l1'],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X, y)

    print(">>> 最佳參數:", grid.best_params_)
    print(">>> 最佳分數:", grid.best_score_)
    return grid.best_estimator_

# === 主程式 ===
if __name__ == "__main__":
    print(">>> backtest_tune.py 啟動")
    df = load_data("txf_5m_day_20251002.csv")
    df = prepare_features(df)
    model = parameter_tuning(df)
    print(">>> GridSearch 完成，模型可用於後續 partial_fit() 更新")
