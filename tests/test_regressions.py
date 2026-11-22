"""Regression tests for bugs found in review. Each test names the bug it guards against."""

import datetime as dt
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from plainquant import indicators
from plainquant.data import CSVProvider, LoadedData, to_ticker
from plainquant.dsl import Strategy
from plainquant.engine import prepare, resolve_session, run
from plainquant.llm import LLMConfig, LLMError, config_from_env, parse_strategy
from plainquant.metrics import compute_metrics
from plainquant.server import create_app


def price_rule(op=">", value=0, left=None):
    return {"type": "compare", "left": left or {"kind": "price"}, "op": op, "right": {"kind": "value", "value": value}}


def strat(*, entry=None, **settings):
    return Strategy.model_validate(
        {
            "name": "r",
            "symbol": "T",
            "capital": 1000,
            "period": {"start": "2025-01-01", "end": "2025-12-31"},
            "entry": {"rules": [entry or price_rule()]},
            "costs": {"slippage_pct": 0},
            **settings,
        }
    )


def frame(rows, times=None, tz="Asia/Kolkata"):
    idx = pd.date_range("2025-01-06", periods=len(rows), tz=tz) if times is None else pd.DatetimeIndex(times, tz=tz)
    data = [(x, x, x, x, 1) if isinstance(x, (int, float)) else (*x, 1) for x in rows]
    return pd.DataFrame(data, index=idx, columns=["open", "high", "low", "close", "volume"])


class Provider:
    def __init__(self, df):
        self.df = df

    def load(self, *args):
        return LoadedData(self.df, "test", [])


def go(df, **settings):
    return run(strat(**settings), Provider(df))


def test_negative_bars_ago_is_rejected():
    with pytest.raises(ValidationError):
        strat(entry=price_rule(value=120, left={"kind": "price", "bars_ago": -1}))


def test_entry_bar_stop_fills_at_open_not_beyond_the_bar():
    for side in ("long", "short"):
        r = go(
            frame([(100, 101, 99, 100)] * 2),
            side=side,
            costs={"slippage_pct": 2},
            risk={"stop_loss": {"unit": "points", "value": 0.5}},
        )
        t = r.trades.iloc[0]
        raw_exit = t.exit_price / (0.98 if side == "long" else 1.02)
        assert raw_exit == pytest.approx(100)


def test_hourly_bar_ending_after_square_off_exits_at_its_open():
    df = frame([100, 100, (100, 130, 100, 130)], ["2025-01-06 13:30", "2025-01-06 14:30", "2025-01-06 15:30"], "America/New_York")
    r = go(df, interval="1h", style="intraday", market="us")
    t = r.trades.iloc[0]
    assert t.exit_reason == "square_off"
    assert t.exit_price == pytest.approx(100)


def test_session_times_must_be_ordered():
    with pytest.raises(ValidationError):
        strat(interval="5m", style="intraday", session={"last_entry": "15:10", "square_off": "15:00"})


def test_full_sizing_leaves_room_for_fees():
    r = go(frame([100, 100, 100]), costs={"slippage_pct": 0, "brokerage_per_order": 5})
    assert r.trades.iloc[0].quantity == 9


def test_orb_crossing_on_the_first_candle_after_the_range():
    df = frame([100, 101, 102, (102, 106, 102, 105), 106, 107], pd.date_range("2025-01-06 09:15", periods=6, freq="5min"))
    orb = {"type": "compare", "left": {"kind": "price"}, "op": "crosses_above", "right": {"kind": "indicator", "name": "orb_high", "minutes": 15}}
    r = go(df, entry=orb, interval="5m", style="intraday")
    assert len(r.trades) == 1
    assert r.trades.iloc[0].entry_time == df.index[4]


def test_orb_range_must_fit_the_candles():
    df = frame([100, 101, 102], pd.date_range("2025-01-06 09:15", periods=3, freq="5min"))
    with pytest.raises(ValueError):
        indicators.orb_high(df, minutes=7)


def test_vwap_is_undefined_without_volume():
    df = frame([100, 120])
    df["volume"] = 0
    assert indicators.vwap(df).isna().all()


def test_indicator_with_no_values_gives_a_clear_error():
    df = frame([100] * 5, pd.date_range("2025-01-06 09:15", periods=5, freq="5min"))
    df["volume"] = 0
    rule = price_rule(">", 0)
    rule["right"] = {"kind": "indicator", "name": "vwap"}
    with pytest.raises(ValueError, match="no volume"):
        go(df, entry=rule, interval="5m", style="intraday")


def test_wilder_atr_is_seeded_with_a_simple_mean():
    df = frame([(100, 101, 100, 100), (100, 110, 100, 100), (100, 110, 100, 100)])
    assert indicators.atr(df, period=3).iloc[-1] == pytest.approx(7)


def test_csv_with_mixed_dst_offsets(tmp_path):
    p = tmp_path / "dst.csv"
    p.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2025-03-07T09:30:00-05:00,100,100,100,100,1\n"
        "2025-03-10T09:30:00-04:00,100,100,100,100,1\n"
    )
    out = CSVProvider(p, timezone="America/New_York").load("T", "5m", dt.date(2025, 3, 7), dt.date(2025, 3, 10), "us")
    assert [t.hour for t in out.frame.index] == [9, 9]


def test_csv_dates_are_filtered_in_the_session_time_zone(tmp_path):
    p = tmp_path / "utc.csv"
    p.write_text("timestamp,open,high,low,close,volume\n2025-01-06T00:30:00Z,100,100,100,100,1\n2025-01-07T00:30:00Z,110,110,110,110,1\n")
    s = strat(
        interval="5m",
        market="us",
        period={"start": "2025-01-06", "end": "2025-01-07"},
        session={"timezone": "America/New_York", "open": "09:30", "close": "23:00", "last_entry": "22:00", "square_off": "22:30"},
    )
    r = run(s, CSVProvider(p))
    assert r.start == "2025-01-06" and r.bars == 1


def test_unfinished_candle_is_dropped():
    s = strat(interval="5m", style="intraday")
    session, _ = resolve_session(s)
    df = frame([100, 101, 102], pd.date_range("2025-01-06 10:00", periods=3, freq="5min"))
    now = pd.Timestamp("2025-01-06 10:12", tz="Asia/Kolkata")
    out, notices = prepare(df, s, session, now=now)
    assert len(out) == 2 and notices[0]["code"] == "unfinished_candle"
    daily = strat()
    out, _ = prepare(frame([100, 101]), daily, resolve_session(daily)[0], now=pd.Timestamp("2025-01-07 11:00", tz="Asia/Kolkata"))
    assert len(out) == 1


def test_sharpe_counts_the_first_day():
    eq = pd.Series([1200.0, 1080.0, 1080.0], index=pd.date_range("2025-01-06", periods=3, tz="Asia/Kolkata"))
    empty = pd.DataFrame(columns=["pnl", "bars_held", "costs"])
    assert compute_metrics(empty, eq, 1000, None)["sharpe"] == pytest.approx(3.46, abs=0.01)


def test_exposure_counts_each_bar_once():
    r = go(frame([100, 100, 100, 100]), exit={"rules": [price_rule()]})
    assert r.metrics["exposure_pct"] == 75.0


def test_tiny_pnl_is_not_rounded_away():
    r = go(
        frame([100, (100, 100.004, 100, 100.004), 100.004, (100.004, 100.004, 100.002, 100.002), 100.002]),
        exit={"rules": [price_rule()]},
        sizing={"method": "fixed_quantity", "value": 1},
    )
    assert r.metrics["profit_factor"] == pytest.approx(2.0)


def test_llm_repairs_out_of_range_params():
    bad = strat().model_dump()
    bad["entry"]["rules"][0]["left"] = {"kind": "indicator", "name": "sma", "period": 0}
    reply = json.dumps({"outcome": "strategy", "strategy": bad})
    with pytest.raises(LLMError):
        parse_strategy("x", LLMConfig("openai", "openai", "m"), _chat=lambda *_: reply)


def test_web_rejects_bad_time_zone_and_infinity():
    client = TestClient(create_app(Provider(frame([100, 100])), LLMConfig("openai", "openai", "m")), raise_server_exceptions=False)
    s = strat().model_dump()
    s["session"] = {"timezone": "Mars/Olympus"}
    assert client.post("/api/explain", json={"strategy": s}).status_code == 400
    assert client.post("/api/backtest", json={"strategy": s}).status_code == 400
    body = '{"strategy":' + strat().model_dump_json().replace('"value":0.0', '"value":Infinity', 1) + "}"
    assert client.post("/api/explain", content=body, headers={"Content-Type": "application/json"}).status_code == 400


def test_hosted_key_is_not_sent_to_a_custom_endpoint(monkeypatch):
    for k in ("PLAINQUANT_API_KEY", "PLAINQUANT_PROVIDER", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "hosted")
    monkeypatch.setenv("PLAINQUANT_MODEL", "m")
    monkeypatch.setenv("PLAINQUANT_BASE_URL", "http://localhost:1234/v1")
    assert config_from_env("openai").api_key is None
    monkeypatch.delenv("PLAINQUANT_BASE_URL")
    assert config_from_env("openai").api_key == "hosted"


def test_hyphenated_nse_symbols_get_the_exchange_suffix():
    assert to_ticker("BAJAJ-AUTO", "india") == "BAJAJ-AUTO.NS"
    assert to_ticker("BTC-USD", "other") == "BTC-USD"


def test_entry_window_after_last_entry_is_sent_back_for_repair():
    from plainquant.llm import semantic_errors

    s = strat(
        interval="5m",
        style="intraday",
        entry={"type": "time_window", "start": "09:20", "end": "09:25"},
        session={"last_entry": "09:20", "square_off": "15:15"},
    )
    assert any("never fill" in e for e in semantic_errors(s))


def test_compact_schema_drops_titles_and_null_branches():
    from plainquant import llm

    text = llm._openai_instructions()
    assert '"title"' not in text and '{"type":"null"}' not in text
    assert len(text) < 13000


def test_daily_trade_times_are_dates():
    from plainquant.report import to_dict

    r = go(frame([100, 101, 102]))
    assert to_dict(r)["trades"][0]["entry_time"] == "2025-01-07"


def test_page_assets_are_bundled_and_allowlisted():
    client = TestClient(create_app(Provider(frame([100, 100])), LLMConfig("openai", "openai", "m")))
    for name, kind in (("Doto.woff2", "font/woff2"), ("GeistMono.woff2", "font/woff2"), ("hero.webp", "image/webp")):
        r = client.get(f"/assets/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == kind and r.content
    assert client.get("/assets/Doto-OFL.txt").status_code == 404
    assert client.get("/assets/..%2Fserver.py").status_code == 404


def test_app_starts_when_the_system_encoding_is_not_utf8(tmp_path):
    """Windows reads files as cp1252 by default; the page and examples contain non-ASCII text."""
    import os
    import subprocess
    import sys

    code = (
        "import locale; assert locale.getpreferredencoding(False).lower() != 'utf-8';"
        "from plainquant import examples; from plainquant.server import create_app;"
        "create_app(provider=object()); [examples.load(n) for n in examples.names()]"
    )
    env = {**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "PLAINQUANT_CONFIG": str(tmp_path / "s.json")}
    r = subprocess.run([sys.executable, "-c", code], check=False, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
