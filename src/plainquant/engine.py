"""The backtest engine.

Execution model (kept simple and conservative on purpose):

- Rules are evaluated on each bar's close. A signal on bar t fills at the open of bar t+1,
  so a strategy never trades on information it could not have had.
- Stops and targets are checked inside each bar using its high and low. If a bar touches
  both, the stop is assumed to have hit first. If a bar opens beyond a level (a gap), the
  fill is at the open.
- Trailing stops follow the best price of the bars before the current one.
- Intraday strategies stop taking entries at `last_entry`, square off at `square_off`, and
  never carry a position overnight.
- One position at a time. Slippage is applied to every fill; brokerage and charges are
  charged per order.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators
from .data import DataProvider, LoadedData
from .dsl import (
    CompareRule,
    Distance,
    IndicatorOperand,
    PriceOperand,
    RuleSet,
    Session,
    Strategy,
    TimeWindowRule,
    WeekdayRule,
)
from .metrics import compute_metrics

SESSIONS = {
    "india": Session(),
    "us": Session(timezone="America/New_York", open="09:30", close="16:00", last_entry="15:30", square_off="15:50"),
    "other": Session(timezone="UTC", open="00:00", close="23:59", last_entry="23:30", square_off="23:45"),
}

_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@dataclass
class BacktestResult:
    strategy: Strategy
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    notices: list[dict]
    data_source: str
    start: str
    end: str
    bars: int
    benchmark_return_pct: float | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- setup


def resolve_session(strategy: Strategy) -> tuple[Session, list[dict]]:
    if strategy.session is not None:
        return strategy.session, []
    s = SESSIONS[strategy.market]
    notice = {
        "code": "assumed_default",
        "message": f"Using {strategy.market} market hours {s.open}-{s.close} ({s.timezone}).",
    }
    return s, [notice] if strategy.interval != "1d" else []


def resolve_period(strategy: Strategy, today: dt.date | None = None) -> tuple[dt.date, dt.date, list[dict]]:
    today = today or dt.date.today()
    p = strategy.period
    end = dt.date.fromisoformat(p.end) if p.end else today
    if p.start:
        start = dt.date.fromisoformat(p.start)
    elif p.lookback_days:
        start = end - dt.timedelta(days=p.lookback_days)
    else:
        start = end - dt.timedelta(days=365)
        return start, end, [{"code": "assumed_default", "message": "No test period given, so the last 1 year is used."}]
    if start >= end:
        raise ValueError("the test period must start before it ends")
    return start, end, []


def prepare(
    df: pd.DataFrame,
    strategy: Strategy,
    session: Session,
    start: dt.date | None = None,
    end: dt.date | None = None,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Put candles in the session time zone, keep the test window, drop candles still forming."""
    notices = []
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(session.timezone)
    else:
        df.index = df.index.tz_convert(session.timezone)
    if start is not None and end is not None:
        d = df.index.date
        df = df[(d >= start) & (d <= end)]
    bar = indicators.INTERVAL_MINUTES[strategy.interval]
    if bar is not None:
        t = df.index.time
        o = dt.time.fromisoformat(session.open)
        c = dt.time.fromisoformat(session.close)
        df = df[(t >= o) & (t < c)]
    now = now if now is not None else pd.Timestamp.now(tz=session.timezone)
    now = now.tz_convert(session.timezone)
    if len(df):
        if bar is not None:
            finished = df.index + pd.Timedelta(minutes=bar) <= now
        else:
            close = dt.time.fromisoformat(session.close)
            last_day = now.date() if now.time() < close else None
            finished = df.index.date != last_day
        if not finished.all():
            notices.append({"code": "unfinished_candle", "message": "The latest candle is still forming, so it was left out."})
            df = df[finished]
    if df.empty:
        raise ValueError("no finished candles inside the trading session and test period")
    return df, notices


# ---------------------------------------------------------------- rule evaluation


class _Evaluator:
    def __init__(self, df: pd.DataFrame, strategy: Strategy, session: Session):
        self.df = df
        self.strategy = strategy
        self.session = session
        self._cache: dict[str, pd.Series] = {}

    def operand(self, op) -> pd.Series | float:
        if isinstance(op, PriceOperand):
            return self.df[op.field].shift(op.bars_ago)
        if isinstance(op, IndicatorOperand):
            spec = indicators.REGISTRY[op.name]
            if spec.intraday_only and self.strategy.interval == "1d":
                raise ValueError(f"{op.name} needs intraday candles, but the strategy uses daily candles")
            params = op.model_dump(include={"period", "fast", "slow", "signal", "std", "multiplier", "minutes"})
            key = f"{op.name}|{sorted(params.items())}|{op.output}"
            if key not in self._cache:
                series = indicators.compute(
                    self.df, op.name, params, op.output, self.session.open,
                    indicators.INTERVAL_MINUTES[self.strategy.interval],
                )
                if series.notna().sum() == 0:
                    hint = " The data has no volume (Yahoo index data has none)." if self.df["volume"].sum() == 0 else ""
                    raise ValueError(
                        f"{op.name} has no values on this data. The test period may be too short for its settings.{hint}"
                    )
                self._cache[key] = series
            return self._cache[key].shift(op.bars_ago)
        return float(op.value)

    def rule(self, r) -> pd.Series:
        idx = self.df.index
        if isinstance(r, CompareRule):
            left, right = self.operand(r.left), self.operand(r.right)
            if r.op in (">", "<", ">=", "<="):
                return _cmp(left, right, r.op, idx)
            above_now = _cmp(left, right, ">", idx)
            below_now = _cmp(left, right, "<", idx)
            lp = left.shift(1) if isinstance(left, pd.Series) else left
            rp = right.shift(1) if isinstance(right, pd.Series) else right
            if r.op == "crosses_above":
                return above_now & _cmp(lp, rp, "<=", idx)
            return below_now & _cmp(lp, rp, ">=", idx)
        if isinstance(r, TimeWindowRule):
            if self.strategy.interval == "1d":
                raise ValueError("time-of-day rules need intraday candles")
            t = pd.Series(idx.time, index=idx)
            s, e = dt.time.fromisoformat(r.start), dt.time.fromisoformat(r.end)
            return (t >= s) & (t < e)
        if isinstance(r, WeekdayRule):
            days = {_WEEKDAYS.index(d) for d in r.days}
            return pd.Series(idx.weekday.isin(list(days)), index=idx)
        raise TypeError(f"unknown rule {r!r}")

    def ruleset(self, rs: RuleSet | None) -> np.ndarray:
        if rs is None:
            return np.zeros(len(self.df), dtype=bool)
        parts = [self.rule(r).fillna(False).astype(bool) for r in rs.rules]
        out = parts[0]
        for p in parts[1:]:
            out = (out & p) if rs.match == "all" else (out | p)
        return out.to_numpy()


def _cmp(a, b, op, idx) -> pd.Series:
    a_s = a if isinstance(a, pd.Series) else pd.Series(a, index=idx)
    b_s = b if isinstance(b, pd.Series) else pd.Series(b, index=idx)
    valid = a_s.notna() & b_s.notna()
    res = {">": a_s > b_s, "<": a_s < b_s, ">=": a_s >= b_s, "<=": a_s <= b_s}[op]
    return res & valid


# ---------------------------------------------------------------- simulation


def simulate(df: pd.DataFrame, strategy: Strategy, session: Session) -> tuple[pd.DataFrame, pd.Series, list[dict]]:
    ev = _Evaluator(df, strategy, session)
    entry_sig = ev.ruleset(strategy.entry)
    exit_sig = ev.ruleset(strategy.exit)
    atr_cache: dict[int, np.ndarray] = {}

    def atr_at(period: int, i: int) -> float:
        if period not in atr_cache:
            atr_cache[period] = indicators.atr(df, period).to_numpy()
        return float(atr_cache[period][i])

    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = df.index
    times = idx.time
    dates = idx.date
    n = len(df)
    intraday = strategy.style == "intraday"
    last_entry = dt.time.fromisoformat(session.last_entry)
    square_off = dt.time.fromisoformat(session.square_off)
    bar_len = pd.Timedelta(minutes=indicators.INTERVAL_MINUTES[strategy.interval] or 0)
    # a candle that ends after square-off can't be held through: exit at its open
    past_square_off = np.array([(ts + bar_len).time() > square_off or ts.time() >= square_off for ts in idx])
    if bar_len:
        past_square_off |= np.array([(ts + bar_len).date() != ts.date() for ts in idx])
    sgn = 1 if strategy.side == "long" else -1
    slip = strategy.costs.slippage_pct / 100
    risk = strategy.risk

    def fees(price: float, qty: int) -> float:
        return strategy.costs.brokerage_per_order + price * qty * strategy.costs.charges_pct / 100

    def affordable(budget: float, fill: float) -> int:
        """Most units whose cost plus entry fees fits in the budget."""
        per_unit = fill * (1 + strategy.costs.charges_pct / 100)
        return max(0, math.floor((budget - strategy.costs.brokerage_per_order) / per_unit))

    def dist(d: Distance, ref: float, i: int, stop_dist: float | None) -> float:
        if d.unit == "percent":
            return ref * d.value / 100
        if d.unit == "points":
            return d.value
        if d.unit == "atr":
            return d.value * atr_at(d.atr_period, i)
        return d.value * (stop_dist or 0.0)

    trades: list[dict] = []
    in_market = np.zeros(n, dtype=bool)
    notices: list[dict] = []
    equity = np.empty(n)
    realized = 0.0
    pos = None  # dict while a position is open
    trades_per_day: dict[dt.date, int] = {}
    skipped_small = 0
    skipped_nan = 0

    def close_position(i: int, raw: float, reason: str):
        nonlocal pos, realized
        fill = raw * (1 - sgn * slip)
        qty = pos["qty"]
        gross = sgn * qty * (raw - pos["raw_entry"])
        slippage_cost = qty * (abs(pos["fill"] - pos["raw_entry"]) + abs(fill - raw))
        fee = pos["entry_fees"] + fees(fill, qty)
        net = sgn * qty * (fill - pos["fill"]) - fee
        realized += net
        trades.append(
            {
                "entry_time": idx[pos["i"]],
                "exit_time": idx[i],
                "side": strategy.side,
                "quantity": qty,
                "entry_price": pos["fill"],
                "exit_price": fill,
                "gross_pnl": gross,
                "costs": slippage_cost + fee,
                "pnl": net,
                "return_pct": 100 * net / (pos["fill"] * qty),
                "bars_held": i - pos["i"] + 1,
                "exit_reason": reason,
            }
        )
        pos = None

    for i in range(n):
        t = times[i]
        if pos is not None:
            in_market[i] = True
        if pos is not None and intraday and past_square_off[i]:
            close_position(i, o[i], "square_off")
        if pos is not None and i > 0 and exit_sig[i - 1]:
            close_position(i, o[i], "exit_rule")

        if pos is None and i > 0 and entry_sig[i - 1]:
            allowed = True
            if intraday and (dates[i] != dates[i - 1] or t >= last_entry or past_square_off[i]):
                allowed = False
            cap = strategy.max_trades_per_day
            if cap is not None and trades_per_day.get(dates[i], 0) >= cap:
                allowed = False
            if allowed:
                raw = o[i]
                fill = raw * (1 + sgn * slip)
                if strategy.sizing.method == "fixed_quantity":
                    qty = int(strategy.sizing.value)
                else:
                    qty = affordable((strategy.capital + realized) * strategy.sizing.value / 100, fill)
                stop_d = dist(risk.stop_loss, fill, i - 1, None) if risk.stop_loss else None
                tp_d = dist(risk.take_profit, fill, i - 1, stop_d) if risk.take_profit else None
                trail_d = dist(risk.trailing_stop, fill, i - 1, None) if risk.trailing_stop else None
                if any(x is not None and not np.isfinite(x) for x in (stop_d, tp_d, trail_d)):
                    skipped_nan += 1
                elif qty < 1:
                    skipped_small += 1
                else:
                    pos = {
                        "i": i,
                        "qty": qty,
                        "raw_entry": raw,
                        "fill": fill,
                        "entry_fees": fees(fill, qty),
                        "stop": fill - sgn * stop_d if stop_d is not None else None,
                        "target": fill + sgn * tp_d if tp_d is not None else None,
                        "trail": trail_d,
                        "best": fill,
                    }
                    trades_per_day[dates[i]] = trades_per_day.get(dates[i], 0) + 1
                    in_market[i] = True

        if pos is not None:
            stop = pos["stop"]
            if pos["trail"] is not None:
                trail_level = pos["best"] - sgn * pos["trail"]
                stop = trail_level if stop is None else (max(stop, trail_level) if sgn == 1 else min(stop, trail_level))
            target = pos["target"]
            hit_stop = stop is not None and (l[i] <= stop if sgn == 1 else h[i] >= stop)
            hit_target = target is not None and (h[i] >= target if sgn == 1 else l[i] <= target)
            # the open is the first tradable price: if it is already past a level, that is the fill
            gapped_stop = stop is not None and (o[i] <= stop if sgn == 1 else o[i] >= stop)
            gapped_target = target is not None and (o[i] >= target if sgn == 1 else o[i] <= target)
            stop_reason = "trailing_stop" if pos["trail"] is not None and stop != pos["stop"] else "stop_loss"
            if gapped_stop:
                close_position(i, o[i], stop_reason)
            elif gapped_target:
                close_position(i, o[i], "take_profit")
            elif hit_stop:
                close_position(i, stop, stop_reason)
            elif hit_target:
                close_position(i, target, "take_profit")

        if pos is not None:
            pos["best"] = max(pos["best"], h[i]) if sgn == 1 else min(pos["best"], l[i])
            last_bar_of_day = i == n - 1 or dates[i + 1] != dates[i]
            if intraday and last_bar_of_day:
                close_position(i, c[i], "session_end")

        unreal = 0.0 if pos is None else sgn * pos["qty"] * (c[i] - pos["fill"]) - pos["entry_fees"]
        equity[i] = strategy.capital + realized + unreal

    if pos is not None:
        close_position(n - 1, c[-1], "end_of_data")
        equity[-1] = strategy.capital + realized

    if skipped_small:
        notices.append(
            {"code": "position_too_small", "message": f"{skipped_small} signals were skipped because capital could not buy one unit."}
        )
    if skipped_nan:
        notices.append(
            {"code": "warmup", "message": f"{skipped_nan} signals were skipped because the ATR for the stop was not ready yet."}
        )
    cols = [
        "entry_time", "exit_time", "side", "quantity", "entry_price", "exit_price",
        "gross_pnl", "costs", "pnl", "return_pct", "bars_held", "exit_reason",
    ]
    trades_df = pd.DataFrame(trades, columns=cols)
    return trades_df, pd.Series(equity, index=idx, name="equity"), in_market, notices


# ---------------------------------------------------------------- top level


def default_notices(strategy: Strategy) -> list[dict]:
    out = []
    s = strategy.model_fields_set
    if "costs" not in s:
        c = strategy.costs
        out.append(
            {
                "code": "assumed_default",
                "message": f"Costs: {c.slippage_pct}% slippage per side, no brokerage or taxes. Set `costs` for your broker.",
            }
        )
    if "sizing" not in s:
        out.append({"code": "assumed_default", "message": "Each trade uses 100% of current capital."})
    if "capital" not in s:
        out.append({"code": "assumed_default", "message": f"Starting capital {strategy.capital:,.0f}."})
    return out


def run(strategy: Strategy, provider: DataProvider, data: LoadedData | None = None) -> BacktestResult:
    session, notices = resolve_session(strategy)
    start, end, period_notices = resolve_period(strategy)
    notices += period_notices + default_notices(strategy)
    if data is None:
        data = provider.load(strategy.symbol, strategy.interval, start, end, strategy.market)
    notices += data.notices
    df, prep_notices = prepare(data.frame, strategy, session, start, end)
    trades, equity, in_market, sim_notices = simulate(df, strategy, session)
    notices += prep_notices + sim_notices
    metrics = compute_metrics(trades, equity, strategy.capital, in_market)
    bench = None
    if len(df) > 1:
        bench = round(100 * (df["close"].iloc[-1] / df["open"].iloc[0] - 1), 2)
        if strategy.side == "short":
            bench = -bench
    return BacktestResult(
        strategy=strategy,
        trades=trades,
        equity=equity,
        metrics=metrics,
        notices=notices,
        data_source=data.source,
        start=str(df.index[0].date()),
        end=str(df.index[-1].date()),
        bars=len(df),
        benchmark_return_pct=bench,
    )
