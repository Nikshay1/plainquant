import numpy as np
import pandas as pd
import pytest

from plainquant import indicators

from .conftest import make_frame


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 300))
    return make_frame(closes, start="2024-01-01", freq="B")


def test_every_indicator_runs_and_has_no_future_leak(df):
    for name, spec in indicators.REGISTRY.items():
        frame = df if not spec.intraday_only else make_frame(df["close"].to_numpy()[:150])
        for out in spec.outputs:
            output = None if out == "value" else out
            full = indicators.compute(frame, name, {}, output, "09:15")
            cut = indicators.compute(frame.iloc[:-20], name, {}, output, "09:15")
            assert len(full) == len(frame)
            pd.testing.assert_series_equal(full.iloc[:-20], cut, check_names=False, rtol=1e-9, atol=1e-9, obj=f"{name}.{out}")


def test_sma_and_ema_values():
    df = make_frame([1, 2, 3, 4, 5], freq="B", start="2024-01-01")
    assert indicators.compute(df, "sma", {"period": 3}, None, "09:15").iloc[-1] == pytest.approx(4)
    ema = indicators.compute(df, "ema", {"period": 3}, None, "09:15")
    assert ema.iloc[-1] == pytest.approx(4.0, abs=0.2)


def test_rsi_bounds(df):
    r = indicators.compute(df, "rsi", {}, None, "09:15").dropna()
    assert r.between(0, 100).all()


def test_highest_excludes_current_bar():
    df = make_frame([1, 2, 3, 10], freq="B", start="2024-01-01", spread=0)
    h = indicators.compute(df, "highest", {"period": 3}, None, "09:15")
    assert h.iloc[-1] == pytest.approx(3)


def test_bad_output_is_rejected(df):
    with pytest.raises(ValueError):
        indicators.compute(df, "macd", {}, "nope", "09:15")
