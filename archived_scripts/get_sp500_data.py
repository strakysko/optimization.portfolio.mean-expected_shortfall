#!/usr/bin/env python3
"""
Pull daily OHLCV data for all S&P 500 constituents from Stooq
and save it as a single compressed CSV.

Requirements:
    pip install pandas requests tqdm lxml

Stooq endpoint pattern:
    https://stooq.com/q/d/l/?s=<ticker>.us&i=d
"""

import io
import os
import time
import gzip
import requests
import pandas as pd
from tqdm import tqdm

# ----------------------------------------------------------------------
# CONFIGURABLE PARAMETERS
# ----------------------------------------------------------------------
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
STOOQ_URL_TEMPLATE = "https://stooq.com/q/d/l/?s={symbol}.us&i=d"
OUT_PATH = "../data/raw/sp500_daily_prices.csv.gz"
REQUEST_TIMEOUT = 20           # seconds per HTTP request
SLEEP_BETWEEN_CALLS = 0.3      # polite delay to avoid hitting Stooq rate limit
# ----------------------------------------------------------------------


def get_sp500_tickers() -> list[str]:
    """Return a list of S&P 500 tickers from Wikipedia (strings, upper-case)."""
    tables = pd.read_html(WIKI_SP500_URL)
    tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).str.upper()
    return tickers.tolist()


def fetch_stooq_csv(symbol: str) -> pd.DataFrame | None:
    """
    Download one ticker from Stooq.
    Returns a DataFrame or None if the request fails.
    """
    url = STOOQ_URL_TEMPLATE.format(symbol=symbol.lower())
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200 or "Exceeded the daily hits limit" in r.text:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        df["Symbol"] = symbol
        return df
    except requests.RequestException:
        return None


def build_sp500_dataset() -> pd.DataFrame:
    """Download and concatenate all tickers into a single DataFrame."""
    tickers = get_sp500_tickers()
    frames = []

    for sym in tqdm(tickers, desc="Downloading", unit="stock"):
        df = fetch_stooq_csv(sym)
        if df is not None and not df.empty:
            frames.append(df)
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not frames:
        raise RuntimeError("No data downloaded — check connectivity or rate limits.")

    # Concatenate and drop exact duplicate rows (if any)
    full_df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["Symbol", "Date"])
        .reset_index(drop=True)
    )
    return full_df


def save_compressed_csv(df: pd.DataFrame, out_path: str = OUT_PATH) -> None:
    """Save DataFrame to a gzip-compressed CSV."""
    # Pandas can write gzip directly, but using gzip.open keeps memory usage low.
    with gzip.open(out_path, "wt", newline="") as gz:
        df.to_csv(gz, index=False)


def main() -> None:
    print("Fetching data …")
    df = build_sp500_dataset()
    print(f"Rows downloaded: {len(df):,}  |  Unique tickers: {df['Symbol'].nunique()}")
    print("Saving compressed CSV …")
    save_compressed_csv(df)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Done. File saved to '{OUT_PATH}' ({size_mb:.1f} MB).")


if __name__ == "__main__":
    main()