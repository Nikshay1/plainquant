# Contributing

Thanks for helping. Issues and pull requests are both welcome.

## Setup

```bash
git clone https://github.com/Nikshay1/plainquant
cd plainquant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
```

The tests use synthetic candles and a fake model, so they need no network and no API key.

## Where things live

| File | What it does |
|---|---|
| `src/plainquant/dsl.py` | The strategy format (Pydantic models) |
| `src/plainquant/indicators.py` | Indicator functions and the registry the model sees |
| `src/plainquant/engine.py` | Signal evaluation and the trade simulator |
| `src/plainquant/metrics.py` | Performance metrics |
| `src/plainquant/llm.py` | Prompt, providers, validation and repair |
| `src/plainquant/data.py` | Yahoo and CSV data providers |
| `src/plainquant/server.py`, `web/index.html` | The local web app |

## Adding an indicator

1. Write a function in `indicators.py` that takes the candle DataFrame and returns a Series.
   It must only use data up to each bar, and `test_every_indicator_runs_and_has_no_future_leak`
   checks this.
2. Register it in `REGISTRY` with its default parameters, outputs and a one-line description.
   The description is what the model reads, so make it clear.
3. Add a test with a known value.

## Adding a data source

Write a class with a `load(symbol, interval, start, end, market)` method that returns
`LoadedData(frame, source_label, notices)`. The frame needs `open`, `high`, `low`, `close` and
`volume` columns and a DatetimeIndex.

## Ground rules

- No look-ahead. A rule evaluated on bar t may only use data up to bar t.
- Keep the strategy format flat. Constrained decoding does not support recursive schemas, and
  small models write flat JSON more reliably.
- Engine behaviour changes need a test in `tests/test_engine.py`.
