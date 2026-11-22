import numpy as np
import pandas as pd
import pytest

from plainquant.data import LoadedData


def make_frame(closes, start="2025-01-06 09:15", freq="5min", tz="Asia/Kolkata", spread=0.5, opens=None, highs=None, lows=None):
    """Candles from a list of closes. open = previous close unless given."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz=tz)
    o = np.asarray(opens, dtype=float) if opens is not None else np.r_[closes[0], closes[:-1]]
    h = np.asarray(highs, dtype=float) if highs is not None else np.maximum(o, closes) + spread
    l = np.asarray(lows, dtype=float) if lows is not None else np.minimum(o, closes) - spread
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": closes, "volume": 1000.0}, index=idx)


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Never read or write the real settings file of whoever runs the tests."""
    monkeypatch.setenv("PLAINQUANT_CONFIG", str(tmp_path / "settings.json"))


class FrameProvider:
    name = "test"

    def __init__(self, frame):
        self.frame = frame

    def load(self, symbol, interval, start, end, market):
        return LoadedData(self.frame, "test data", [])


@pytest.fixture
def daily_trend():
    # down, then up, then down: one golden-cross style trade
    closes = list(np.linspace(120, 100, 40)) + list(np.linspace(100, 140, 40)) + list(np.linspace(140, 110, 40))
    return make_frame(closes, start="2024-01-01", freq="B")
