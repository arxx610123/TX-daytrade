#!/usr/bin/env python3
"""
convert_txf5m_batch_v251005f.py
批次轉換 TXF RAW 檔 -> 5 分鐘K線檔（只取當天日盤 08:45~13:45）
輸入: TXF_RAW_TO_5M/RAW_ALL/*.csv
輸出: data/5m_day/txf_5m_day_YYYYMMDD.csv
格式: 同 txf_5m_day_20251002.csv
"""

import re
import pandas as pd
from pathlib import Path

ENCODINGS = ["utf-8", "utf-8-sig", "cp950", "latin1"]
TARGET_COLUMNS = ["datetime", "Open", "High", "Low", "Close", "Volume"]

def project_root():
    return Path(__file__).resolve().parent.parent

def default_paths():
    root = project_root()
    return root / "TXF_RAW_TO_5M" / "RAW_ALL", root / "data" / "5m_day"

def read_csv_flex(path: Path):
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"{path.name}: cannot read with common encodings")

def detect_datetime(df, fname):
    cols_lower = [c.lower() for c in df.columns]

    # case 1: date + time
    if "date" in cols_lower and "time" in cols_lower:
        dcol = df.columns[cols_lower.index("date")]
        tcol = df.columns[cols_lower.index("time")]
        return pd.to_datetime(df[dcol].astype(str) + " " + df[tcol].astype(str),
                              errors="coerce", infer_datetime_format=True)

    # case 2: 中文 成交日期 + 成交時間
    if "成交日期" in df.columns and "成交時間" in df.columns:
        return pd.to_datetime(df["成交日期"].astype(str) + " " + df["成交時間"].astype(str),
                              errors="coerce", infer_datetime_format=True)

    # case 3: 其他 datetime-like 欄位
    for c in df.columns:
        if any(k in c.lower() for k in ["datetime", "time", "date", "時間", "日期"]):
            s = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            if s.notna().any():
                return s

    raise ValueError(f"{fname}: No valid datetime column found")

def detect_cols(df):
    price = next((c for c in df.columns if "price" in c.lower() or "成交價" in c), None)
    vol   = next((c for c in df.columns if "vol" in c.lower() or "量" in c), None)
    if not price or not vol:
        raise ValueError("Missing price/volume col")
    return price, vol

def convert_file(path: Path, outdir: Path, symbol="txf"):
    df = read_csv_flex(path)
    df["__dt__"] = detect_datetime(df, path.name)
    price, vol = detect_cols(df)

    df = df.dropna(subset=["__dt__"]).set_index("__dt__")
    df[price] = pd.to_numeric(df[price], errors="coerce")
    df[vol]   = pd.to_numeric(df[vol], errors="coerce").fillna(0)

    # 從檔名解析日期
    m = re.search(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})", path.name)
    if not m:
        raise ValueError(f"Cannot parse date from filename {path.name}")
    ymd = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    target_date = pd.to_datetime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")

    # ✅ 只取當天日盤 (08:45–13:45)
    df = df[df.index.date == target_date.date()]
    df = df.between_time("08:45", "13:45")

    # Resample 成 5 分鐘
    ohlc = df[price].resample("5T", label="right", closed="right").ohlc()
    vol5 = df[vol].resample("5T", label="right", closed="right").sum()
    out  = pd.concat([ohlc, vol5], axis=1).dropna(subset=["open"])
    out.columns = ["Open", "High", "Low", "Close", "Volume"]

    out = out.reset_index()
    out["datetime"] = out["__dt__"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out[TARGET_COLUMNS]

    # 輸出檔名
    out_path = outdir / f"{symbol}_5m_day_{ymd}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"SUCCESS {path.name} -> {out_path} ({len(out)} rows)")

def main():
    indir, outdir = default_paths()
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(indir.glob("*.csv"))
    if not files:
        print(f"NO CSV found in {indir}")
        return

    total, success, fail = 0, 0, 0
    for f in files:
        total += 1
        try:
            convert_file(f, outdir)
            success += 1
        except Exception as e:
            fail += 1
            print(f"ERROR {f.name}: {e}")

    print(f"SUMMARY total={total}, success={success}, fail={fail}")

if __name__ == "__main__":
    main()
