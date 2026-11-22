import pandas as pd
import pytest

from plainquant.dsl import Strategy
from plainquant.engine import run

from .conftest import FrameProvider, make_frame


def ema_cross(**over):
    base = {
        "name": "t",
        "symbol": "TEST",
        "interval": "1d",
        "period": {"start": "2020-01-01", "end": "2030-01-01"},
        "entry": {"rules": [{"type": "compare", "left": {"kind": "indicator", "name": "ema", "period": 3}, "op": "crosses_above", "right": {"kind": "indicator", "name": "ema", "period": 8}}]},
        "exit": {"rules": [{"type": "compare", "left": {"kind": "indicator", "name": "ema", "period": 3}, "op": "crosses_below", "right": {"kind": "indicator", "name": "ema", "period": 8}}]},
        "costs": {"slippage_pct": 0},
        "sizing": {"method": "fixed_quantity", "value": 1},
    }
    base.update(over)
    return Strategy.model_validate(base)


def price_above(level, **over):
    base = {
        "name": "t",
        "symbol": "TEST",
        "interval": "5m",
        "style": "intraday",
        "period": {"start": "2020-01-01", "end": "2030-01-01"},
        "entry": {"rules": [{"type": "compare", "left": {"kind": "price", "field": "close"}, "op": ">", "right": {"kind": "value", "value": level}}]},
        "costs": {"slippage_pct": 0},
        "sizing": {"method": "fixed_quantity", "value": 10},
    }
    base.update(over)
    return Strategy.model_validate(base)


def test_crossover_round_trip(daily_trend):
    r = run(ema_cross(), FrameProvider(daily_trend))
    assert len(r.trades) >= 1
    t = r.trades.iloc[0]
    assert t.exit_reason == "exit_rule"
    assert t.pnl > 0
    assert r.metrics["trades"] == len(r.trades)


def test_signal_fills_at_next_open():
    closes = [100] * 5 + [106] + [107] * 4
    df = make_frame(closes)
    r = run(price_above(105, style="positional", interval="5m"), FrameProvider(df))
    t = r.trades.iloc[0]
    # signal on bar 5 (close 106), fill at bar 6 open == previous close 106
    assert t.entry_time == df.index[6]
    assert t.entry_price == pytest.approx(df["open"].iloc[6])


def test_stop_loss_hits_inside_bar():
    closes = [100, 100, 101, 101, 99, 99]
    lows = [99.5, 99.5, 100.5, 100.5, 95.0, 98.5]
    df = make_frame(closes, lows=lows)
    s = price_above(100.5, style="positional", risk={"stop_loss": {"unit": "percent", "value": 2}})
    r = run(s, FrameProvider(df))
    t = r.trades.iloc[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == pytest.approx(101 * 0.98)


def test_gap_through_stop_fills_at_open():
    closes = [100, 101, 101, 90, 90]
    opens = [100, 100, 101, 92, 90]
    df = make_frame(closes, opens=opens)
    s = price_above(100.5, style="positional", risk={"stop_loss": {"unit": "percent", "value": 2}})
    r = run(s, FrameProvider(df))
    t = r.trades.iloc[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == pytest.approx(92)


def test_stop_wins_when_bar_hits_both():
    closes = [100, 101, 101, 101]
    highs = [100.5, 101.5, 101.5, 110]
    lows = [99.5, 100.5, 100.5, 90]
    df = make_frame(closes, highs=highs, lows=lows)
    s = price_above(100.5, style="positional", risk={"stop_loss": {"unit": "percent", "value": 2}, "take_profit": {"unit": "percent", "value": 2}})
    r = run(s, FrameProvider(df))
    assert r.trades.iloc[0].exit_reason == "stop_loss"


def test_take_profit_in_risk_multiples():
    closes = [100, 101, 101, 101, 101]
    highs = [100.5, 101.5, 101.5, 105, 101.5]
    df = make_frame(closes, highs=highs)
    s = price_above(100.5, style="positional", risk={"stop_loss": {"unit": "points", "value": 1}, "take_profit": {"unit": "rr", "value": 2}})
    r = run(s, FrameProvider(df))
    t = r.trades.iloc[0]
    assert t.exit_reason == "take_profit"
    assert t.exit_price == pytest.approx(103)


def test_trailing_stop_follows_price():
    closes = [100, 101, 105, 110, 109, 103, 103]
    df = make_frame(closes, spread=0.2)
    s = price_above(100.5, style="positional", risk={"trailing_stop": {"unit": "percent", "value": 5}})
    r = run(s, FrameProvider(df))
    t = r.trades.iloc[0]
    assert t.exit_reason == "trailing_stop"
    assert t.pnl > 0


def test_intraday_squares_off_and_never_holds_overnight():
    day1 = make_frame([100] + [101] * 77, start="2025-01-06 09:15")
    day2 = make_frame([101] * 75, start="2025-01-07 09:15")
    df = pd.concat([day1, day2])
    s = price_above(100.5, max_trades_per_day=1)
    r = run(s, FrameProvider(df))
    assert len(r.trades) == 2
    for t in r.trades.itertuples():
        assert t.entry_time.date() == t.exit_time.date()
        assert t.exit_reason in ("square_off", "session_end")
        assert t.exit_time.time() >= pd.Timestamp("15:15").time()


def test_intraday_no_entries_after_cutoff():
    closes = [100] * 70 + [101] * 5
    df = make_frame(closes, start="2025-01-06 09:15")  # first 101 close at 15:05
    r = run(price_above(100.5), FrameProvider(df))
    assert len(r.trades) == 0


def test_costs_reduce_pnl(daily_trend):
    free = run(ema_cross(), FrameProvider(daily_trend))
    paid = run(ema_cross(costs={"slippage_pct": 0.1, "brokerage_per_order": 20}), FrameProvider(daily_trend))
    assert paid.metrics["net_pnl"] < free.metrics["net_pnl"]
    assert paid.metrics["total_costs"] > 0


def test_percent_of_capital_sizing(daily_trend):
    s = ema_cross(sizing={"method": "percent_of_capital", "value": 50}, capital=10000)
    r = run(s, FrameProvider(daily_trend))
    t = r.trades.iloc[0]
    assert t.quantity == int(5000 // t.entry_price)


def test_short_side(daily_trend):
    s = ema_cross(side="short")
    r = run(s, FrameProvider(daily_trend))
    assert (r.trades["side"] == "short").all()


def test_orb_waits_for_range_to_finish():
    closes = [100, 102, 101] + [101.5] * 3 + [104] + [104] * 10
    df = make_frame(closes, start="2025-01-06 09:15", spread=0.1)
    s = Strategy.model_validate(
        {
            "name": "orb",
            "symbol": "T",
            "interval": "5m",
            "style": "intraday",
            "period": {"start": "2020-01-01", "end": "2030-01-01"},
            "entry": {"rules": [{"type": "compare", "left": {"kind": "price"}, "op": "crosses_above", "right": {"kind": "indicator", "name": "orb_high", "minutes": 15}}]},
            "costs": {"slippage_pct": 0},
        }
    )
    r = run(s, FrameProvider(df))
    assert len(r.trades) == 1
    assert r.trades.iloc[0].entry_time == df.index[7]


def test_time_rule_on_daily_is_rejected(daily_trend):
    s = ema_cross()
    s.entry.rules.append(__import__("plainquant.dsl", fromlist=["TimeWindowRule"]).TimeWindowRule(type="time_window", start="09:30", end="10:00"))
    with pytest.raises(ValueError):
        run(s, FrameProvider(daily_trend))


def test_default_notices(daily_trend):
    s = Strategy.model_validate({**ema_cross().model_dump(exclude={"costs", "sizing", "capital"})})
    r = run(s, FrameProvider(daily_trend))
    text = " ".join(n["message"] for n in r.notices)
    assert "slippage" in text and "capital" in text
