"""
data.py
=======
Fetching market data:
  - Stock prices via yfinance (AAPL)
  - Rate curve via FRED (US Treasury yields, proxy for risk-free rate)
  - Historical volatility computation
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime

# ─── FRED series ID → maturity in years ───────────────────────────────────────
FRED_SERIES = {
    "DGS1MO":  1/12,
    "DGS3MO":  3/12,
    "DGS6MO":  6/12,
    "DGS1":    1.0,
    "DGS2":    2.0,
    "DGS3":    3.0,
    "DGS5":    5.0,
    "DGS7":    7.0,
    "DGS10":  10.0,
    "DGS20":  20.0,
    "DGS30":  30.0,
}

# Fallback flat curve (4%) used when FRED is unreachable
_FALLBACK_RATE = 0.04


def fetch_rate_curve(as_of: datetime = None) -> dict:
    """
    Download the latest US Treasury zero-coupon yields from FRED.

    Returns
    -------
    dict  {maturity_in_years: rate_in_decimal}
          e.g. {0.083: 0.053, 0.25: 0.054, ..., 30.0: 0.048}
    """
    if as_of is None:
        as_of = datetime.today()

    rates = {}
    for series_id, mat in FRED_SERIES.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df = pd.read_csv(url, parse_dates=["DATE"])
            df = df[df["VALUE"] != "."].copy()
            df["VALUE"] = df["VALUE"].astype(float) / 100.0   # % → decimal
            df = df.sort_values("DATE")
            subset = df[df["DATE"] <= pd.Timestamp(as_of)]
            if not subset.empty:
                rates[mat] = float(subset.iloc[-1]["VALUE"])
        except Exception:
            pass  # will use fallback below

    if len(rates) < 3:
        # Fallback: slightly upward-sloping curve
        rates = {
            1/12: 0.053, 3/12: 0.054, 6/12: 0.053,
            1.0:  0.050, 2.0:  0.047, 3.0:  0.046,
            5.0:  0.045, 7.0:  0.045, 10.0: 0.044,
            20.0: 0.045, 30.0: 0.044,
        }

    return rates


def fetch_stock_history(ticker: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    """
    Download OHLCV history for *ticker* from yfinance.

    Returns
    -------
    pd.DataFrame  with columns [Open, High, Low, Close, Volume]
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist.index = hist.index.tz_localize(None)   # remove timezone for simplicity
    return hist[["Open", "High", "Low", "Close", "Volume"]]


def get_spot_price(ticker: str = "AAPL") -> float:
    """Return the latest available closing price."""
    hist = fetch_stock_history(ticker, period="5d")
    return float(hist["Close"].iloc[-1])


def get_dividend_yield(ticker: str = "AAPL") -> float:
    """Return the annualized dividend yield (decimal). Defaults to 0 if unavailable."""
    try:
        info = yf.Ticker(ticker).info
        return float(info.get("dividendYield") or 0.0)
    except Exception:
        return 0.0


def historical_volatility(ticker: str = "AAPL", window: int = 252, period: str = "2y") -> float:
    """
    Compute annualized historical volatility from daily log-returns.

    Parameters
    ----------
    window : int   Number of trading days in a year (for annualisation)
    period : str   yfinance period string, e.g. "2y", "1y"

    Returns
    -------
    float   Annualized volatility in decimal form (e.g. 0.28 = 28%)
    """
    hist = fetch_stock_history(ticker, period)
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(window))


def vol_term_structure(ticker: str = "AAPL", periods_months: list = None) -> dict:
    """
    Estimate a simple volatility term-structure by computing realized vol
    over different rolling windows.

    Returns
    -------
    dict  {maturity_in_years: realized_vol}
    """
    if periods_months is None:
        periods_months = [1, 3, 6, 12, 24]

    hist = fetch_stock_history(ticker, period="3y")
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()

    term_struct = {}
    for m in periods_months:
        n_days = int(m * 21)          # ~21 trading days per month
        subset = log_ret.iloc[-n_days:]
        vol = float(subset.std() * np.sqrt(252))
        term_struct[m / 12] = vol     # key = maturity in years

    return term_struct
