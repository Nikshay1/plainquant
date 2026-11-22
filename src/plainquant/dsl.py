"""The strategy format.

A strategy is plain JSON. It is deliberately flat (no nested condition trees) so that
small local models can write it reliably and so it works with JSON-schema constrained
decoding, which does not support recursive schemas.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .indicators import INDICATOR_NAMES

IndicatorName = Literal[tuple(INDICATOR_NAMES)]  # type: ignore[valid-type]

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- operands


class PriceOperand(_Model):
    """A raw candle value, e.g. the close of the current bar."""

    kind: Literal["price"]
    field: Literal["open", "high", "low", "close", "volume"] = "close"
    bars_ago: int = Field(0, description="0 = current bar, 1 = previous bar, ...")


class IndicatorOperand(_Model):
    """An indicator value. Only the parameters the indicator uses are read."""

    kind: Literal["indicator"]
    name: IndicatorName  # type: ignore[valid-type]
    period: int | None = None
    fast: int | None = None
    slow: int | None = None
    signal: int | None = None
    std: float | None = None
    multiplier: float | None = None
    minutes: int | None = None
    output: str | None = Field(None, description="Which line of a multi-line indicator, e.g. 'signal' for MACD")
    bars_ago: int = 0


class ValueOperand(_Model):
    """A constant number, e.g. 70 for an RSI level."""

    kind: Literal["value"]
    value: float


Operand = PriceOperand | IndicatorOperand | ValueOperand


# ---------------------------------------------------------------- rules


class CompareRule(_Model):
    type: Literal["compare"]
    left: Operand
    op: Literal[">", "<", ">=", "<=", "crosses_above", "crosses_below"]
    right: Operand


class TimeWindowRule(_Model):
    """True when the bar starts inside [start, end) in the session time zone."""

    type: Literal["time_window"]
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not _HHMM.match(v):
            raise ValueError("time must be HH:MM (24h)")
        return v


class WeekdayRule(_Model):
    type: Literal["weekday"]
    days: list[Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]]


Rule = CompareRule | TimeWindowRule | WeekdayRule


class RuleSet(_Model):
    """`all`: every rule must hold. `any`: at least one rule must hold."""

    match: Literal["all", "any"] = "all"
    rules: list[Rule]

    @field_validator("rules")
    @classmethod
    def _non_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("a rule set needs at least one rule")
        return v


# ---------------------------------------------------------------- risk, sizing, costs


class Distance(_Model):
    """How far a stop or target sits from the entry price.

    percent: % of entry price. points: absolute price points.
    atr: multiple of ATR(atr_period) at entry. rr: multiple of the stop distance
    (take-profit only, needs a stop loss).
    """

    unit: Literal["percent", "points", "atr", "rr"]
    value: float
    atr_period: int = 14

    @field_validator("value")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("distance must be positive")
        return v


class Risk(_Model):
    stop_loss: Distance | None = None
    take_profit: Distance | None = None
    trailing_stop: Distance | None = None

    @model_validator(mode="after")
    def _check(self) -> Risk:
        for name in ("stop_loss", "trailing_stop"):
            d = getattr(self, name)
            if d is not None and d.unit == "rr":
                raise ValueError(f"{name} cannot use unit 'rr'")
        if self.take_profit is not None and self.take_profit.unit == "rr" and self.stop_loss is None:
            raise ValueError("take_profit in 'rr' needs a stop_loss")
        return self


class Sizing(_Model):
    method: Literal["fixed_quantity", "percent_of_capital"] = "percent_of_capital"
    value: float = 100.0


class Costs(_Model):
    slippage_pct: float = Field(0.05, description="Per side, % of price")
    brokerage_per_order: float = Field(0.0, description="Flat amount per order")
    charges_pct: float = Field(0.0, description="Taxes and exchange charges per side, % of turnover")


class Period(_Model):
    lookback_days: int | None = Field(None, description="Test the last N calendar days")
    start: str | None = Field(None, description="YYYY-MM-DD")
    end: str | None = Field(None, description="YYYY-MM-DD, defaults to today")


class Session(_Model):
    timezone: str = "Asia/Kolkata"
    open: str = "09:15"
    close: str = "15:30"
    last_entry: str = Field("15:00", description="Intraday: no new entries from this time")
    square_off: str = Field("15:15", description="Intraday: close open positions at this time")

    @field_validator("open", "close", "last_entry", "square_off")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not _HHMM.match(v):
            raise ValueError("time must be HH:MM (24h)")
        return v


# ---------------------------------------------------------------- strategy


class Strategy(_Model):
    name: str
    symbol: str = Field(description="Ticker, e.g. RELIANCE, NIFTY, BANKNIFTY, AAPL, or a raw Yahoo ticker like TCS.NS")
    market: Literal["india", "us", "other"] = "india"
    side: Literal["long", "short"] = "long"
    interval: Literal["1m", "5m", "15m", "30m", "1h", "1d"] = "1d"
    style: Literal["intraday", "positional"] = "positional"
    period: Period = Field(default_factory=Period)
    entry: RuleSet
    exit: RuleSet | None = None
    risk: Risk = Field(default_factory=Risk)
    sizing: Sizing = Field(default_factory=Sizing)
    capital: float = 100_000.0
    costs: Costs = Field(default_factory=Costs)
    session: Session | None = None
    max_trades_per_day: int | None = None

    @model_validator(mode="after")
    def _check(self) -> Strategy:
        if self.style == "intraday" and self.interval == "1d":
            raise ValueError("an intraday strategy needs an intraday interval (1m to 1h)")
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        return self


# ---------------------------------------------------------------- LLM output

DeclineCode = Literal[
    "options_or_futures",
    "multiple_instruments",
    "multiple_timeframes",
    "fundamentals",
    "unsupported_indicator",
    "unsupported_logic",
    "unsupported_data",
    "look_ahead",
    "too_vague",
    "not_a_strategy",
]


class Assumption(_Model):
    field: str = Field(description="Which setting was filled in, e.g. 'interval'")
    value: str = Field(description="The value used")
    reason: str = Field(description="Why, in one short sentence")


class ParseResult(_Model):
    """What the model returns for one prompt."""

    outcome: Literal["strategy", "decline"]
    strategy: Strategy | None = None
    assumptions: list[Assumption] = Field(default_factory=list)
    decline_code: DeclineCode | None = None
    decline_reason: str | None = None
    suggestion: str | None = Field(None, description="When declining, a close version that can be tested")

    @model_validator(mode="after")
    def _check(self) -> ParseResult:
        if self.outcome == "strategy" and self.strategy is None:
            raise ValueError("outcome 'strategy' needs a strategy")
        if self.outcome == "decline" and not self.decline_reason:
            raise ValueError("outcome 'decline' needs a decline_reason")
        return self
