"""Market data.

Two providers ship with plainquant:

- YahooProvider: free, no key, covers NSE/BSE stocks and indices, US stocks and more.
  Yahoo limits intraday history (1m: 7 days, 5m-30m: 60 days, 1h: 730 days).
- CSVProvider: your own candles, e.g. exported from your broker. No limits.

Anything with a `load(symbol, interval, start, end, market)` method that returns an OHLCV
DataFrame works as a provider, so broker APIs can be plugged in the same way.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]

INDIA_ALIASES = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "NIFTY BANK": "^NSEBANK",
    "BANK NIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "NIFTYIT": "^CNXIT",
    "NIFTY IT": "^CNXIT",
    "SENSEX": "^BSESN",
    "INDIAVIX": "^INDIAVIX",
    "INDIA VIX": "^INDIAVIX",
}

# Furthest back Yahoo serves each interval, in calendar days.
YAHOO_LIMITS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "1h": 729, "1d": None}


@dataclass
class LoadedData:
    frame: pd.DataFrame
    source: str
    notices: list[dict]


class DataProvider(Protocol):
    name: str

    def load(self, symbol: str, interval: str, start: dt.date, end: dt.date, market: str) -> LoadedData: ...


def to_ticker(symbol: str, market: str) -> str:
    s = symbol.strip().upper()
    if market == "india":
        if s in INDIA_ALIASES:
            return INDIA_ALIASES[s]
        if any(c in s for c in ".^="):
            return s
        return f"{s.replace(' ', '')}.NS"
    return s


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"data is missing columns: {', '.join(missing)}")
    df = df[COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


class YahooProvider:
    name = "yahoo"

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "plainquant"

    def load(self, symbol, interval, start, end, market) -> LoadedData:
        import yfinance as yf

        notices = []
        ticker = to_ticker(symbol, market)
        limit = YAHOO_LIMITS[interval]
        today = dt.date.today()
        if limit is not None and (today - start).days > limit:
            new_start = today - dt.timedelta(days=limit)
            notices.append(
                {
                    "code": "period_clipped",
                    "message": (
                        f"Yahoo only serves {limit} days of {interval} candles, so the test starts on "
                        f"{new_start} instead of {start}. Use a CSV file from your broker for longer intraday history."
                    ),
                }
            )
            start = new_start
            if start > end:
                raise ValueError(
                    f"Yahoo only keeps the last {limit} days of {interval} candles, and this test period ended earlier. "
                    "Use daily candles or a CSV file from your broker."
                )
        cached = self._cache_path(ticker, interval, start, end)
        if cached.exists() and end < today:
            df = pd.read_pickle(cached)
        else:
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + dt.timedelta(days=1)).isoformat(),
                interval=interval,
                auto_adjust=False,  # traded prices (split-adjusted), so levels match your charts
                progress=False,
                multi_level_index=False,
            )
            if raw is None or raw.empty:
                raise ValueError(f"No data from Yahoo for {ticker} ({interval}, {start} to {end}). Check the symbol.")
            df = normalise(raw)
            if end < today:
                cached.parent.mkdir(parents=True, exist_ok=True)
                df.to_pickle(cached)
        return LoadedData(df, f"Yahoo Finance ({ticker})", notices)

    def _cache_path(self, ticker, interval, start, end) -> Path:
        key = hashlib.sha1(f"{ticker}|{interval}|{start}|{end}".encode()).hexdigest()[:16]
        return self.cache_dir / f"{key}.pkl"


class CSVProvider:
    """Reads candles from a CSV with columns timestamp (or date/datetime), open, high, low, close, volume.

    Timestamps without a time zone are taken to be in `timezone`.
    """

    name = "csv"

    def __init__(self, path: str | Path, timezone: str | None = None):
        self.path = Path(path)
        self.timezone = timezone

    def load(self, symbol, interval, start, end, market) -> LoadedData:
        df = pd.read_csv(self.path)
        df.columns = [c.strip().lower() for c in df.columns]
        tcol = next((c for c in ("timestamp", "datetime", "date", "time") if c in df.columns), None)
        if tcol is None:
            raise ValueError("CSV needs a timestamp, datetime or date column")
        raw = df.pop(tcol)
        try:
            stamps = pd.to_datetime(raw)
        except ValueError:
            stamps = None
        if stamps is None or not (isinstance(stamps.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_dtype(stamps)):
            stamps = pd.to_datetime(raw, utc=True)  # mixed UTC offsets, e.g. across a DST change
        df.index = pd.DatetimeIndex(stamps)
        df = normalise(df)
        if self.timezone:
            tz = self.timezone
            df.index = df.index.tz_localize(tz) if df.index.tz is None else df.index.tz_convert(tz)
        # coarse window with a day of margin; the engine trims to exact session dates
        lo = pd.Timestamp(start) - pd.Timedelta(days=1)
        hi = pd.Timestamp(end) + pd.Timedelta(days=2)
        if df.index.tz is not None:
            lo, hi = lo.tz_localize(df.index.tz), hi.tz_localize(df.index.tz)
        df = df[(df.index >= lo) & (df.index < hi)]
        if df.empty:
            raise ValueError(f"{self.path} has no candles between {start} and {end}")
        return LoadedData(df, f"CSV ({self.path.name})", [])
