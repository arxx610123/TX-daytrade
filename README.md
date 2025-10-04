# TX-daytrade

滾動式學習模型（5分鐘K線）

## 更新紀錄
### v1.1 (2025-10-04)
- 加入 SimpleImputer，自動處理技術指標計算後產生的 NaN。
- technical_indicators.py 增加 inf→NaN 清理。
- pipeline 增加標準化 StandardScaler。
- 已通過 Colab 測試，可進行 GridSearch 調參與模型訓練。

## 執行方式
```bash
!rm -rf TX-daytrade
!git clone https://github.com/<你的帳號>/TX-daytrade.git
%cd TX-daytrade
!python backtest_tune.py
