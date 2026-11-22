# Roadmap

One step at a time, in this order. Each step ends with tests passing, the README updated, and a
commit. A step is checked off only after it has been run for real, not just unit-tested.

## Step 0: Dry run with a real model (done 2026-11-22)
- [x] Run the whole project end to end on a real model. GPT-6 Luna wasn't available (no OpenAI
      key), so this used Groq `openai/gpt-oss-120b`, plus `gpt-oss-20b` and `qwen3.8-27b` for a
      small-model check.
- [x] CLI: `parse`, `run` and `backtest` with 16 prompts covering simple, intraday, short, ATR
      stops, ORB, timed entries and Hinglish, plus declines for options, pairs, fundamentals,
      mixed timeframes, nested logic and vague prompts. Final result: 16/16 correct.
- [x] Web app: describe, check rules, run, results and the invalid-JSON guard, driven in headless
      Chromium.
- [x] Fixed from the dry run:
      - Prompt was too large for Groq's free tier (8k tokens/min): compact schema, about 40% smaller,
        and more retries on 429.
      - Model picked positional for 15-minute candles: stronger intraday guideline.
      - Golden cross had no exit: named setups get their usual exit.
      - "Buy at 9:20" produced 0 trades (model set last_entry to 09:20, window one candle late):
        timed-entry guideline and a validation check.
      - "LT" not recognised as a ticker: ticker guideline.
      - Small models invented time or weekday rules and changed the candle size: stricter "don't add
        rules" guideline.
      - UI: daily trades showed 00:00, old results stayed after an error, missing favicon.
- [ ] Re-run on GPT-6 Luna once an OpenAI key is available.

## Step 1: Release polish
- [ ] Screenshot or GIF of the web app in the README.
- [ ] Publish to PyPI so `pip install plainquant` works.
- [ ] Tag v0.2.0 with release notes.

## Step 2: Nested logic
- [ ] Support `(A and B) or C` without a recursive schema, e.g. a list of rule groups
      combined with `any`, where each group is `all`.
- [ ] Update the prompt, explainer and web app. Stop declining `unsupported_logic` for these cases.

## Step 3: Several instruments
- [ ] Signals from one symbol, trades on another (e.g. NIFTY trend filter, trade a stock).
- [ ] Run the same strategy over a list of stocks and compare the results.
- [ ] Portfolio of several positions with shared capital.

## Step 4: Futures
- [ ] Futures data source (continuous contracts with roll handling) through a CSV or broker
      provider, since Yahoo has no Indian futures.
- [ ] Lot sizes, margin and rollover costs.

## Step 5: Options
- [ ] Options chain data provider (historical option prices are not free; plug in a paid or
      broker source).
- [ ] Legs (CE/PE, buy/sell), strike selection (ATM offset, premium, delta), expiry selection.
- [ ] Multi-leg structures (straddle, strangle, iron condor), leg-level and combined stops.

## Later ideas
- Broker data providers (Zerodha Kite, Upstox, Dhan) for long intraday history.
- Parameter sweeps ("try EMA 5 to 30").
- Walk-forward / out-of-sample split to catch overfitting.
