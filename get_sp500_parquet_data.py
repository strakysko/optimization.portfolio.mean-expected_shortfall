#!/usr/bin/env python3
"""
Pull daily Close prices for all S&P 500 constituents from Stooq
and save a space-efficient Parquet file.

Requirements:
    pip install pandas pyarrow requests tqdm lxml
"""

import io
import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIGURABLE PARAMETERS
# ----------------------------------------------------------------------
WIKI_SP500_URL      = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
STOOQ_URL_TEMPLATE  = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"
OUT_PATH            = "data/raw/sp500_daily_close.parquet"
REQUEST_TIMEOUT     = 20        # seconds per HTTP request
SLEEP_BETWEEN_CALLS = 0.3       # polite delay between calls
# ----------------------------------------------------------------------


def get_sp500_tickers() -> list[str]:
    """Return a list of S&P 500 tickers (upper-case, '.'→'-')."""
    tables = pd.read_html(WIKI_SP500_URL)
    return (
        tables[0]["Symbol"]
        .str.replace(".", "-", regex=False)
        .str.upper()
        .tolist()
    )


def fetch_stooq_csv(symbol: str) -> pd.DataFrame | None:
    """Download one ticker from Stooq, return DataFrame with Date & Close."""
    url = STOOQ_URL_TEMPLATE.format(symbol=symbol.lower())
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200 or "Exceeded the daily hits limit" in r.text:
            print("Status code:", r.status_code)
            print("Response (first 300 chars):", r.text[:300])
            return None
        df = pd.read_csv(io.StringIO(r.text), usecols=["Date", "Close"])
        df["Symbol"] = symbol            # add ticker column
        return df
    except requests.RequestException:
        return None


def build_dataset() -> pd.DataFrame:
    """Download all tickers and concatenate."""
    frames = []
    for sym in tqdm(get_sp500_tickers(), desc="Downloading", unit="stock"):
        df = fetch_stooq_csv(sym)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not frames:
        raise RuntimeError("No data downloaded — check connectivity or rate limits.")

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["Symbol", "Date"])
        .reset_index(drop=True)
    )

    # ──►  Memory-efficient dtypes
    df["Date"]   = pd.to_datetime(df["Date"])       # datetime64[ns]
    df["Symbol"] = df["Symbol"].astype("category")  # dictionary encoding
    df["Close"]  = df["Close"].astype("float32")    # ½ the bytes of float64

    return df


def save_parquet(df: pd.DataFrame, out_path: str = OUT_PATH) -> None:
    """Write Parquet with Snappy compression."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, compression="snappy", engine="pyarrow", index=False)


def main() -> None:
    print("Fetching data …")
    df = build_dataset()
    print(f"Rows downloaded: {len(df):,} | Unique tickers: {df['Symbol'].nunique()}")
    print("Saving Parquet …")
    save_parquet(df)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Done. File saved to '{OUT_PATH}' ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()
