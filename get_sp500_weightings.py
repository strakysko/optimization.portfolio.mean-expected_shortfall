#!/usr/bin/env python3
"""
Extract SPY constituent tickers and index weights into a Pandas DataFrame.

Requirements
------------
pip install pandas requests openpyxl
"""

import io
import requests
import pandas as pd

# --- Config --------------------------------------------------------------- #
URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
    "holdings-daily-us-en-spy.xlsx"
)
TICKER_COL_CANDIDATES = {"ticker", "ticker symbol", "symbol"}
WEIGHT_COL_CANDIDATES = {"weight (%)", "weight", "weight_percent"}
NORMALISE_WEIGHT = False   # True → 0-1 scale, False → leave in %
# -------------------------------------------------------------------------- #

def download_excel(url: str) -> bytes:
    """Fetch the Excel file and return its raw bytes."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content

def load_spy_weights(url: str = URL, normalise: bool = False, skip_first: int = 4, rows_to_read: int = 504) -> pd.DataFrame:
    """Return a DataFrame with two columns: ['ticker', 'weight']."""
    raw_bytes = download_excel(url)

    df = pd.read_excel(
        io.BytesIO(raw_bytes),
        sheet_name=0,
        skiprows=skip_first,
        nrows=rows_to_read,
        engine="openpyxl",
    )[["Ticker", "Weight"]]

    # Match Ticker styling to Symbol stylin in sp500 price histories dataset
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)

    # detele row corresponding to the weighting of US Dollars in the ETF
    df = df[df["Ticker"] != "-"]

    if normalise:
        df["Weight"] = df["Weight"] / 100.0

    # Cast Ticker to category with 16 bits
    df["Ticker"] = df["Ticker"].astype("category")

    # Reset index for tidy output
    return df.reset_index(drop=True)

if __name__ == "__main__":
    spy_df = load_spy_weights(normalise=NORMALISE_WEIGHT)
    print(spy_df.head())        # quick sanity-check
    # Save to CSV in data/raw directory
    output_path = "data/raw/sp500_weightings.csv"
    spy_df.to_csv(output_path, index=False)
    print(f"SP500 weightings saved to {output_path}")
    # If you need the DataFrame in other code, just import load_spy_weights().
