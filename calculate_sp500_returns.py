#!/usr/bin/env python3
"""
Calculate rolling 10-day returns for S&P 500 constituents

Requirements
------------
pip install pandas numpy
"""

import pandas as pd
import numpy as np


def load_data():
    """Load S&P 500 weights and price history data."""
    # Load S&P 500 weights
    weights_df = pd.read_csv('data/raw/sp500_weightings.csv')

    # Load daily close prices (last 756 trading days)
    prices_df = pd.read_parquet('data/raw/sp500_daily_close.parquet')

    # Ensure we only use the last 756 trading days
    unique_dates = prices_df['Date'].unique()
    unique_dates = sorted(unique_dates, reverse=True)[:756]
    prices_df = prices_df[prices_df['Date'].isin(unique_dates)]

    return weights_df, prices_df


def calculate_rolling_returns(prices_df, window=10):
    """Calculate rolling returns over specified window of trading days."""
    # Create a pivot table of prices: dates as index, symbols as columns
    pivot_df = prices_df.pivot(index='Date', columns='Symbol', values='Close')

    # Sort by date (ascending) to ensure correct return calculation
    pivot_df = pivot_df.sort_index()

    # Calculate rolling returns (percentage change over window days)
    # Formula: (price_t / price_{t-window} - 1)
    rolling_returns = (pivot_df / pivot_df.shift(window) - 1)

    # Reset index to prepare for melting
    rolling_returns = rolling_returns.reset_index()

    # Melt the dataframe to get it back to long format
    melted_returns = pd.melt(
        rolling_returns,
        id_vars=['Date'],
        var_name='Symbol',
        value_name='Return_10d'
    )

    return melted_returns


def main():
    """Main function to execute the script."""
    # Load data
    weights_df, prices_df = load_data()

    # Calculate rolling 10-day returns
    returns_df = calculate_rolling_returns(prices_df, window=10)

    # Convert weights to a dictionary for faster lookup
    weights_dict = dict(zip(weights_df['Ticker'], weights_df['Weight']))

    # Multiply returns by corresponding weights
    returns_df['Weighted_Return_10d'] = returns_df.apply(
        lambda row: row['Return_10d'] * weights_dict.get(row['Symbol'], 0),
        axis=1
    )

    # Drop nulls from returns (first window days will be NaN)
    returns_df = returns_df.dropna(subset=['Return_10d'])

    # Save results
    output_path = 'data/processed/sp500_10day_returns.parquet'
    returns_df.to_parquet(output_path, index=False)
    print(f"Calculated 10-day rolling returns saved to {output_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total tickers processed: {returns_df['Symbol'].nunique()}")
    print(f"Date range: {returns_df['Date'].min()} to {returns_df['Date'].max()}")
    print(f"Average weighted 10-day return: {returns_df['Weighted_Return_10d'].mean():.6f}%")
    print(f"Average unweighted 10-day return: {returns_df['Return_10d'].mean():.4f}%")

    print(returns_df.head())

    return returns_df


if __name__ == "__main__":
    result_df = main()
