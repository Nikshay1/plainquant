"""Turn a strategy back into plain English so people can check it before running it."""

from __future__ import annotations

from .dsl import (
    CompareRule,
    Distance,
    IndicatorOperand,
    PriceOperand,
    RuleSet,
    Strategy,
    TimeWindowRule,
    ValueOperand,
    WeekdayRule,
)
from .indicators import REGISTRY

_OPS = {
    ">": "is above",
    "<": "is below",
    ">=": "is at or above",
    "<=": "is at or below",
    "crosses_above": "crosses above",
    "crosses_below": "crosses below",
}

_INTERVALS = {"1m": "1-minute", "5m": "5-minute", "15m": "15-minute", "30m": "30-minute", "1h": "1-hour", "1d": "daily"}


def _ago(n: int) -> str:
    return "" if n == 0 else (" on the previous bar" if n == 1 else f" {n} bars ago")


def operand(op) -> str:
    if isinstance(op, ValueOperand):
        v = op.value
        return str(int(v)) if v == int(v) else str(v)
    if isinstance(op, PriceOperand):
        return f"{op.field}{_ago(op.bars_ago)}"
    assert isinstance(op, IndicatorOperand)
    spec = REGISTRY[op.name]
    args = []
    for k in ("period", "fast", "slow", "signal", "std", "multiplier", "minutes"):
        v = getattr(op, k)
        if v is None and k in spec.params:
            v = spec.params[k]
        if v is not None and k in spec.params:
            args.append(str(int(v)) if float(v) == int(v) else str(v))
    name = op.name.upper() if len(op.name) <= 4 else op.name.replace("_", " ")
    s = f"{name}({', '.join(args)})" if args else name
    if op.output and len(spec.outputs) > 1:
        s += f" {op.output.replace('_', ' ')}"
    return s + _ago(op.bars_ago)


def rule(r) -> str:
    if isinstance(r, CompareRule):
        return f"{operand(r.left)} {_OPS[r.op]} {operand(r.right)}"
    if isinstance(r, TimeWindowRule):
        return f"time is between {r.start} and {r.end}"
    if isinstance(r, WeekdayRule):
        return "day is " + ", ".join(d.capitalize() for d in r.days)
    raise TypeError(r)


def ruleset(rs: RuleSet) -> str:
    parts = [rule(r) for r in rs.rules]
    if len(parts) == 1:
        return parts[0]
    joiner = " AND " if rs.match == "all" else " OR "
    return joiner.join(parts)


def distance(d: Distance) -> str:
    if d.unit == "percent":
        return f"{d.value:g}%"
    if d.unit == "points":
        return f"{d.value:g} points"
    if d.unit == "atr":
        return f"{d.value:g}x ATR({d.atr_period})"
    return f"{d.value:g}x the stop distance"


def explain(s: Strategy) -> list[str]:
    side = "Buy" if s.side == "long" else "Sell short"
    lines = [
        f"Instrument: {s.symbol} ({s.market}), {_INTERVALS[s.interval]} candles, {s.style}",
        f"Entry: {side} at the next bar's open when {ruleset(s.entry)}",
    ]
    if s.exit:
        lines.append(f"Exit: at the next bar's open when {ruleset(s.exit)}")
    if s.risk.stop_loss:
        lines.append(f"Stop loss: {distance(s.risk.stop_loss)} from entry")
    if s.risk.take_profit:
        lines.append(f"Target: {distance(s.risk.take_profit)} from entry")
    if s.risk.trailing_stop:
        lines.append(f"Trailing stop: {distance(s.risk.trailing_stop)} behind the best price")
    if not (s.exit or s.risk.stop_loss or s.risk.take_profit or s.risk.trailing_stop or s.style == "intraday"):
        lines.append("Exit: none set, so a trade is held until the end of the test")
    if s.style == "intraday":
        lines.append("Intraday: positions are squared off before the close")
    if s.sizing.method == "fixed_quantity":
        lines.append(f"Size: {s.sizing.value:g} units per trade")
    else:
        lines.append(f"Size: {s.sizing.value:g}% of current capital per trade")
    p = s.period
    if p.start:
        lines.append(f"Period: {p.start} to {p.end or 'today'}")
    elif p.lookback_days:
        lines.append(f"Period: last {p.lookback_days} days")
    return lines
