import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from technical_indicators import add_technical_indicators

# === 資料讀取與清理 ===
def load_and_clean_data(file_path):
    print(">>> 開始讀取資料")
    df = pd.read_csv(file_path)
    print(f">>> 成功讀取 {len(df)} 筆資料，欄位: {list(df.columns)}")
    df = add_technical_indicators(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# === Grid Search 調參 ===
def parameter_tuning(X, y):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("sgd", SGDClassifier(max_iter=1000, tol=1e-3, random_state=42))
    ])

    param_grid = {
        "sgd__loss": ["hinge", "log_loss", "modified_huber"],
        "sgd__alpha": [0.0001, 0.001, 0.01],
        "sgd__penalty": ["l2", "l1", "elasticnet"]
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    print(">>> 最佳參數:", grid.best_params_)
    return grid.best_estimator_

# === 簡易回測 ===
def backtest(model, X, y):
    preds = model.predict(X)
    accuracy = np.mean(preds == y)
    print(f">>> Backtest Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    data_path = "data/5m_day/txf_5m_day_20251002.csv"
    df = load_and_clean_data(data_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    model = parameter_tuning(X, y)
    backtest(model, X, y)

