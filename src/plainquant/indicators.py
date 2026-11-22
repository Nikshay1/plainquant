"""Indicator library.

Every indicator takes an OHLCV DataFrame (columns open, high, low, close, volume, with a
tz-aware DatetimeIndex in the session time zone) and returns a Series aligned to it.
Values at bar t only use data up to and including bar t, so nothing looks ahead.

To add an indicator: write a function, then register it in REGISTRY with its defaults,
outputs and a one-line description. The description is shown to the LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Spec:
    fn: Callable[..., pd.Series]
    description: str
    params: dict = field(default_factory=dict)  # name -> default
    outputs: tuple[str, ...] = ("value",)
    intraday_only: bool = False


# ---------------------------------------------------------------- helpers


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)


def _dates(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df.index.date, index=df.index)


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    d = _dates(df)
    return df.groupby(d.values).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))


# ---------------------------------------------------------------- moving averages


def sma(df, period=20, **_):
    return df["close"].rolling(period, min_periods=period).mean()


def ema(df, period=20, **_):
    return _ema(df["close"], period)


def wma(df, period=20, **_):
    w = np.arange(1, period + 1, dtype=float)
    return df["close"].rolling(period, min_periods=period).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def vwap(df, **_):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    d = _dates(df).values
    pv = (tp * df["volume"]).groupby(d).cumsum()
    vol = df["volume"].groupby(d).cumsum()
    return (pv / vol.replace(0, np.nan)).fillna(tp)


def volume_sma(df, period=20, **_):
    return df["volume"].rolling(period, min_periods=period).mean()


# ---------------------------------------------------------------- momentum


def rsi(df, period=14, **_):
    delta = df["close"].diff()
    gain = _wilder(delta.clip(lower=0), period)
    loss = _wilder(-delta.clip(upper=0), period)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(df, fast=12, slow=26, signal=9, output="macd", **_):
    line = _ema(df["close"], fast) - _ema(df["close"], slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return {"macd": line, "signal": sig, "hist": line - sig}[output or "macd"]


def stochastic(df, period=14, signal=3, output="k", **_):
    ll = df["low"].rolling(period, min_periods=period).min()
    hh = df["high"].rolling(period, min_periods=period).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    k = k.rolling(3, min_periods=3).mean()  # slow %K
    d = k.rolling(signal, min_periods=signal).mean()
    return {"k": k, "d": d}[output or "k"]


def roc(df, period=10, **_):
    return 100 * (df["close"] / df["close"].shift(period) - 1)


def cci(df, period=20, **_):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period, min_periods=period).mean()
    md = tp.rolling(period, min_periods=period).apply(lambda x: float(np.mean(np.abs(x - x.mean()))), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(df, period=14, **_):
    hh = df["high"].rolling(period, min_periods=period).max()
    ll = df["low"].rolling(period, min_periods=period).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def adx(df, period=14, output="adx", **_):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_ = _wilder(_true_range(df), period)
    plus_di = 100 * _wilder(plus_dm, period) / atr_
    minus_di = 100 * _wilder(minus_dm, period) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return {"adx": _wilder(dx, period), "plus_di": plus_di, "minus_di": minus_di}[output or "adx"]


def obv(df, **_):
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


# ---------------------------------------------------------------- volatility and trend


def atr(df, period=14, **_):
    return _wilder(_true_range(df), period)


def bollinger(df, period=20, std=2.0, output="middle", **_):
    mid = df["close"].rolling(period, min_periods=period).mean()
    sd = df["close"].rolling(period, min_periods=period).std(ddof=0)
    upper, lower = mid + std * sd, mid - std * sd
    return {
        "middle": mid,
        "upper": upper,
        "lower": lower,
        "width": (upper - lower) / mid,
        "percent_b": (df["close"] - lower) / (upper - lower).replace(0, np.nan),
    }[output or "middle"]


def supertrend(df, period=10, multiplier=3.0, output="line", **_):
    atr_ = atr(df, period).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)
    upper = hl2 + multiplier * atr_
    lower = hl2 - multiplier * atr_
    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    fu, fl = np.nan, np.nan
    d = 1
    for i in range(n):
        if np.isnan(atr_[i]):
            continue
        if np.isnan(fu):
            fu, fl = upper[i], lower[i]
        else:
            fu = upper[i] if (upper[i] < fu or close[i - 1] > fu) else fu
            fl = lower[i] if (lower[i] > fl or close[i - 1] < fl) else fl
            if d == 1 and close[i] < fl:
                d = -1
            elif d == -1 and close[i] > fu:
                d = 1
        line[i] = fl if d == 1 else fu
        direction[i] = d
    s = {"line": line, "direction": direction}[output or "line"]
    return pd.Series(s, index=df.index)


def highest(df, period=20, **_):
    """Highest high of the previous `period` bars (excludes the current bar)."""
    return df["high"].rolling(period, min_periods=period).max().shift(1)


def lowest(df, period=20, **_):
    """Lowest low of the previous `period` bars (excludes the current bar)."""
    return df["low"].rolling(period, min_periods=period).min().shift(1)


# ---------------------------------------------------------------- session levels


def day_open(df, **_):
    return df["open"].groupby(_dates(df).values).transform("first")


def _prev_day(df, col):
    daily = _daily(df)[col].shift(1)
    return pd.Series(_dates(df).map(daily).to_numpy(dtype=float), index=df.index)


def prev_day_high(df, **_):
    return _prev_day(df, "high")


def prev_day_low(df, **_):
    return _prev_day(df, "low")


def prev_day_close(df, **_):
    return _prev_day(df, "close")


def _orb(df, minutes, session_open, col, agg):
    t0 = pd.Timestamp(f"2000-01-01 {session_open}")
    cutoff = (t0 + pd.Timedelta(minutes=minutes)).time()
    times = pd.Series(df.index.time, index=df.index)
    in_range = times < cutoff
    level = df[col].where(in_range).groupby(_dates(df).values).transform(agg)
    return level.where(~in_range)  # unknown until the range has finished


def orb_high(df, minutes=15, session_open="09:15", **_):
    return _orb(df, minutes, session_open, "high", "max")


def orb_low(df, minutes=15, session_open="09:15", **_):
    return _orb(df, minutes, session_open, "low", "min")


def gap_pct(df, **_):
    """Today's open vs yesterday's close, in %."""
    return 100 * (day_open(df) / prev_day_close(df) - 1)


# ---------------------------------------------------------------- registry

REGISTRY: dict[str, Spec] = {
    "sma": Spec(sma, "Simple moving average of close", {"period": 20}),
    "ema": Spec(ema, "Exponential moving average of close", {"period": 20}),
    "wma": Spec(wma, "Weighted moving average of close", {"period": 20}),
    "vwap": Spec(vwap, "Volume-weighted average price, resets each session", {}, intraday_only=True),
    "volume_sma": Spec(volume_sma, "Average volume", {"period": 20}),
    "rsi": Spec(rsi, "Relative strength index (Wilder), 0 to 100", {"period": 14}),
    "macd": Spec(macd, "MACD", {"fast": 12, "slow": 26, "signal": 9}, ("macd", "signal", "hist")),
    "stochastic": Spec(stochastic, "Slow stochastic oscillator, 0 to 100", {"period": 14, "signal": 3}, ("k", "d")),
    "roc": Spec(roc, "Rate of change in %", {"period": 10}),
    "cci": Spec(cci, "Commodity channel index", {"period": 20}),
    "williams_r": Spec(williams_r, "Williams %R, -100 to 0", {"period": 14}),
    "adx": Spec(adx, "Average directional index and DI lines", {"period": 14}, ("adx", "plus_di", "minus_di")),
    "obv": Spec(obv, "On-balance volume", {}),
    "atr": Spec(atr, "Average true range (Wilder)", {"period": 14}),
    "bollinger": Spec(
        bollinger, "Bollinger bands", {"period": 20, "std": 2.0}, ("middle", "upper", "lower", "width", "percent_b")
    ),
    "supertrend": Spec(
        supertrend,
        "Supertrend. output 'line' is the stop line, 'direction' is 1 (up) or -1 (down)",
        {"period": 10, "multiplier": 3.0},
        ("line", "direction"),
    ),
    "highest": Spec(highest, "Highest high of the previous N bars (breakout level)", {"period": 20}),
    "lowest": Spec(lowest, "Lowest low of the previous N bars (breakdown level)", {"period": 20}),
    "day_open": Spec(day_open, "Today's opening price", {}, intraday_only=True),
    "prev_day_high": Spec(prev_day_high, "Previous session high", {}),
    "prev_day_low": Spec(prev_day_low, "Previous session low", {}),
    "prev_day_close": Spec(prev_day_close, "Previous session close", {}),
    "orb_high": Spec(orb_high, "Opening range high of the first N minutes (known once the range ends)", {"minutes": 15}, intraday_only=True),
    "orb_low": Spec(orb_low, "Opening range low of the first N minutes (known once the range ends)", {"minutes": 15}, intraday_only=True),
    "gap_pct": Spec(gap_pct, "Today's open vs previous close, in %", {}, intraday_only=True),
}

INDICATOR_NAMES: list[str] = list(REGISTRY)

_PARAM_NAMES = ("period", "fast", "slow", "signal", "std", "multiplier", "minutes")


def compute(df: pd.DataFrame, name: str, params: dict, output: str | None, session_open: str) -> pd.Series:
    spec = REGISTRY[name]
    kwargs = dict(spec.params)
    for k in _PARAM_NAMES:
        if params.get(k) is not None:
            kwargs[k] = params[k]
    if output is not None:
        if output not in spec.outputs:
            raise ValueError(f"{name} has no output '{output}'. Choose from {', '.join(spec.outputs)}")
        if len(spec.outputs) > 1:
            kwargs["output"] = output
    for k in ("period", "fast", "slow", "signal", "minutes"):
        if k in kwargs and int(kwargs[k]) < 1:
            raise ValueError(f"{name}: {k} must be at least 1")
        if k in kwargs:
            kwargs[k] = int(kwargs[k])
    return spec.fn(df, session_open=session_open, **kwargs)


def describe() -> str:
    """A compact table of indicators for prompts and docs."""
    lines = []
    for name, spec in REGISTRY.items():
        params = ", ".join(f"{k}={v}" for k, v in spec.params.items()) or "none"
        outputs = "" if spec.outputs == ("value",) else f"; outputs: {', '.join(spec.outputs)}"
        note = "; intraday only" if spec.intraday_only else ""
        lines.append(f"- {name}: {spec.description} (params: {params}{outputs}{note})")
    return "\n".join(lines)
