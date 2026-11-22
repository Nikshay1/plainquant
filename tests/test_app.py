import json

import numpy as np
from fastapi.testclient import TestClient

from plainquant import examples
from plainquant.cli import main
from plainquant.dsl import Strategy
from plainquant.engine import run
from plainquant.report import to_dict, to_html, to_text
from plainquant.server import create_app

from .conftest import FrameProvider, make_frame


def frame():
    closes = list(np.linspace(120, 100, 60)) + list(np.linspace(100, 150, 80)) + list(np.linspace(150, 120, 60))
    return make_frame(closes, start="2024-01-01", freq="B")


def test_examples_are_valid():
    assert len(examples.names()) >= 4
    for name in examples.names():
        assert isinstance(examples.load(name), Strategy)


def test_reports_render():
    raw = examples.load("golden_cross_nifty").model_dump()
    raw["period"] = {"start": "2020-01-01", "end": "2030-01-01"}
    s = Strategy.model_validate(raw)
    r = run(s, FrameProvider(frame()))
    d = to_dict(r)
    json.dumps(d)
    assert "metrics" in d and "rules" in d
    assert "<svg" in to_html(r) or "curve(" in to_html(r)
    assert "Net P&L" in to_text(r)


def test_server_flow():
    client = TestClient(create_app(provider=FrameProvider(frame())))
    assert "plainquant" in client.get("/").text
    exs = client.get("/api/examples").json()
    strat = next(e for e in exs if e["id"] == "rsi_reversal_reliance")["strategy"]
    strat["period"] = {"start": "2020-01-01", "end": "2030-01-01"}
    r = client.post("/api/backtest", json={"strategy": strat})
    assert r.status_code == 200, r.text
    assert "metrics" in r.json()
    bad = client.post("/api/backtest", json={"strategy": {"name": "x"}})
    assert bad.status_code == 400


def test_cli_backtest_csv(tmp_path, capsys):
    df = frame()
    csv = tmp_path / "c.csv"
    df.tz_localize(None).rename_axis("date").to_csv(csv)
    s = examples.load("supertrend_tcs").model_dump(exclude_none=True)
    s["period"] = {"start": "2020-01-01", "end": "2030-01-01"}
    f = tmp_path / "s.json"
    f.write_text(json.dumps(s))
    out_html = tmp_path / "r.html"
    assert main(["backtest", str(f), "--csv", str(csv), "--html", str(out_html)]) == 0
    assert out_html.exists()
    assert "Return %" in capsys.readouterr().out
