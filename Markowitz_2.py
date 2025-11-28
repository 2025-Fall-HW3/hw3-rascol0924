"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        # 1. Calculate the rolling cumulative return (Momentum)
        # Use min_periods=lookback to ensure calculation only starts when a full window is available.
        rolling_cum_return = (1 + self.returns[assets]).rolling(
            window=self.lookback, 
            min_periods=self.lookback
        ).apply(np.prod, raw=True) - 1
        
        # 2. Rank the assets based on momentum for each day
        # axis=1 ranks assets row-wise; ascending=False gives rank 1 to the best performer.
        ranked_assets = rolling_cum_return.rank(axis=1, ascending=False)
        
        # 3. Create a boolean mask for the top N performers
        # True if the asset's rank is less than or equal to the desired number of holdings (top_n)
        top_n_mask = (ranked_assets <= self.top_n)
        
        # 4. Calculate the weight value and initialize the weights DataFrame
        weight_value = 1.0 / self.top_n
        momentum_weights = pd.DataFrame(0.0, index=self.price.index, columns=assets)
        
        # 5. Assign the weight value using the boolean mask (vectorized assignment)
        momentum_weights[top_n_mask] = weight_value
        
        # 6. Apply the shift to avoid lookahead bias: weights calculated on day t are used on day t+1.
        momentum_weights_shifted = momentum_weights.shift(1)
        
        # 7. Assign the shifted weights to the portfolio_weights DataFrame
        self.portfolio_weights.loc[:, assets] = momentum_weights_shifted

        """
        TODO: Complete Task 4 Above
        """
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
