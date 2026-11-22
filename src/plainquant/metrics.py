"""Performance metrics computed from the trade list and the equity curve."""

from __future__ import annotations

import math

import pandas as pd


def _r(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd)


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, capital: float, interval: str) -> dict:
    daily = equity.groupby(equity.index.date).last()
    daily.index = pd.to_datetime(daily.index)
    rets = daily.pct_change().dropna()

    peak = equity.cummax()
    dd = equity - peak
    dd_pct = dd / peak
    final = float(equity.iloc[-1]) if len(equity) else capital
    days = max((daily.index[-1] - daily.index[0]).days, 1) if len(daily) > 1 else 1
    years = days / 365.25
    cagr = (final / capital) ** (1 / years) - 1 if final > 0 and years >= 0.25 else None

    std = rets.std(ddof=1) if len(rets) > 1 else float("nan")
    sharpe = rets.mean() / std * math.sqrt(252) if std and std > 0 else None
    downside = rets[rets < 0]
    dstd = math.sqrt((downside**2).mean()) if len(downside) else float("nan")
    sortino = rets.mean() / dstd * math.sqrt(252) if dstd and dstd > 0 else None

    n = len(trades)
    wins = trades[trades["pnl"] > 0] if n else trades
    losses = trades[trades["pnl"] <= 0] if n else trades
    gross_win = float(wins["pnl"].sum()) if n else 0.0
    gross_loss = float(-losses["pnl"].sum()) if n else 0.0
    max_dd_pct = float(dd_pct.min()) * 100 if len(dd_pct) else 0.0

    # longest losing streak
    streak = best_streak = 0
    for p in trades["pnl"] if n else []:
        streak = streak + 1 if p <= 0 else 0
        best_streak = max(best_streak, streak)

    exposure = None
    if n and len(equity):
        held = int(trades["bars_held"].sum())
        exposure = 100 * min(held, len(equity)) / len(equity)

    return {
        "net_pnl": _r(final - capital),
        "net_return_pct": _r(100 * (final / capital - 1)),
        "final_equity": _r(final),
        "cagr_pct": _r(cagr * 100) if cagr is not None else None,
        "max_drawdown_pct": _r(max_dd_pct),
        "max_drawdown": _r(float(dd.min()) if len(dd) else 0.0),
        "sharpe": _r(sharpe),
        "sortino": _r(sortino),
        "calmar": _r((cagr * 100) / abs(max_dd_pct)) if cagr is not None and max_dd_pct < 0 else None,
        "trades": n,
        "win_rate_pct": _r(100 * len(wins) / n) if n else None,
        "profit_factor": _r(gross_win / gross_loss) if gross_loss > 0 else None,
        "expectancy": _r(trades["pnl"].mean()) if n else None,
        "avg_win": _r(wins["pnl"].mean()) if len(wins) else None,
        "avg_loss": _r(losses["pnl"].mean()) if len(losses) else None,
        "best_trade": _r(trades["pnl"].max()) if n else None,
        "worst_trade": _r(trades["pnl"].min()) if n else None,
        "max_losing_streak": best_streak,
        "avg_bars_held": _r(trades["bars_held"].mean(), 1) if n else None,
        "exposure_pct": _r(exposure, 1),
        "total_costs": _r(trades["costs"].sum()) if n else 0.0,
        "trading_days": len(daily),
    }
