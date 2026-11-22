"""Turn a plain-English strategy into a validated Strategy, using the user's own model.

Bring your own inference. Supported out of the box:

- Claude via the official Anthropic SDK (ANTHROPIC_API_KEY)
- Any OpenAI-compatible endpoint: OpenAI, Ollama, LM Studio, OpenRouter, Groq, vLLM, ...

Nothing here talks to a plainquant server. Your prompt goes only to the endpoint you configure.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from pydantic import ValidationError

from . import indicators
from .dsl import CompareRule, IndicatorOperand, ParseResult, Strategy, TimeWindowRule

PRESETS: dict[str, dict] = {
    "anthropic": {"kind": "anthropic", "model": "claude-opus-5-5", "key_env": "ANTHROPIC_API_KEY"},
    "openai": {"kind": "openai", "base_url": None, "key_env": "OPENAI_API_KEY"},
    "ollama": {"kind": "openai", "base_url": "http://localhost:11434/v1", "key_env": None},
    "lmstudio": {"kind": "openai", "base_url": "http://localhost:1234/v1", "key_env": None},
    "openrouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
}

# Models that accept the server-side refusal fallback.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-fable-5-2"}


class LLMNotConfigured(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str
    kind: str
    model: str
    base_url: str | None = None
    api_key: str | None = None

    def label(self) -> str:
        where = f" at {self.base_url}" if self.base_url else ""
        return f"{self.provider}: {self.model}{where}"


SETUP_HELP = """No model is configured. plainquant uses your own model, pick one:

  Claude:        export ANTHROPIC_API_KEY=...
  OpenAI:        export OPENAI_API_KEY=...  PLAINQUANT_MODEL=<model>
  Ollama (free, local):
                 ollama pull <model>
                 export PLAINQUANT_PROVIDER=ollama PLAINQUANT_MODEL=<model>
  Any OpenAI-compatible server:
                 export PLAINQUANT_PROVIDER=openai PLAINQUANT_BASE_URL=<url> PLAINQUANT_MODEL=<model>

Or skip the model and run a strategy file: plainquant backtest strategy.json"""


def config_from_env(provider: str | None = None, model: str | None = None, base_url: str | None = None) -> LLMConfig:
    env = os.environ
    provider = provider or env.get("PLAINQUANT_PROVIDER")
    if not provider:
        if env.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif env.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise LLMNotConfigured(SETUP_HELP)
    provider = provider.lower()
    if provider not in PRESETS:
        raise LLMNotConfigured(f"Unknown provider '{provider}'. Choose from: {', '.join(PRESETS)}")
    preset = PRESETS[provider]
    model = model or env.get("PLAINQUANT_MODEL") or preset.get("model")
    if not model:
        raise LLMNotConfigured(f"Set a model for {provider}: --model <name> or PLAINQUANT_MODEL=<name>")
    base_url = base_url or env.get("PLAINQUANT_BASE_URL") or preset.get("base_url")
    key_env = preset.get("key_env")
    # A provider's own key is only sent to that provider's endpoint. Custom endpoints use PLAINQUANT_API_KEY.
    custom_endpoint = base_url != preset.get("base_url")
    api_key = env.get("PLAINQUANT_API_KEY") or (env.get(key_env) if key_env and not custom_endpoint else None)
    return LLMConfig(provider=provider, kind=preset["kind"], model=model, base_url=base_url, api_key=api_key)


# ---------------------------------------------------------------- prompt

SYSTEM_PROMPT = f"""You convert a trader's plain-English strategy into a JSON strategy for the plainquant backtester.

How the engine works:
- One instrument per strategy, long or short. Rules are checked on each candle's close and orders fill at the next candle's open.
- Data comes from Yahoo Finance: Indian stocks and indices (NSE/BSE), US stocks, and other Yahoo tickers.
- Entry and exit are rule sets: `match` is "all" (AND) or "any" (OR) over a flat list of rules. Nesting is not supported.
- Rule types: "compare" (left op right, op is >, <, >=, <=, crosses_above, crosses_below), "time_window" (HH:MM start inclusive, end exclusive) and "weekday".
- Operands: {{"kind":"price","field":"close"}}, {{"kind":"indicator","name":...,"period":...}}, {{"kind":"value","value":70}}. Any operand can set bars_ago (1 = previous candle).
- Risk exits: stop_loss, take_profit, trailing_stop, each {{"unit": "percent"|"points"|"atr", "value": number}}. take_profit can also use unit "rr" (multiple of the stop distance).
- Intraday strategies (style "intraday") take no entries after 15:00 and square off at 15:15 in India. Positional strategies hold overnight.

Indicators (use only these names; only set the params listed):
{indicators.describe()}

Guidance:
- Symbols: write NSE names as the trader says them (RELIANCE, TCS, HDFCBANK). Indices: NIFTY, BANKNIFTY, FINNIFTY, SENSEX. Set market to "us" for US tickers.
- "X crosses above Y" -> crosses_above. "RSI oversold bounce/reversal" -> rsi crosses_above 30 (long) or rsi crosses_below 70 (short).
- Breakouts: "close above the 20-day high" -> close > highest(20). highest/lowest already exclude the current candle.
- Opening range breakout: close crosses_above orb_high with minutes = the range length.
- "Supertrend turns up" -> supertrend output "direction" crosses_above value 0.
- Periods: "last 1 year" -> period.lookback_days 365; "5 years" -> 1825; exact dates -> start/end (YYYY-MM-DD).
- Candles: "5-min" -> interval "5m". Use style "intraday" for intraday candles unless the trader says to hold overnight or positional.
- Fill in missing settings with sensible defaults (daily candles, last 1 year, long side, 100% of capital per trade, 100000 capital) and list EVERY value you chose that the trader did not state in `assumptions`.
- Keep the trader's numbers exactly. Do not add stops, targets or filters they did not ask for.
- Give the strategy a short descriptive name.

Decline (outcome "decline") when the strategy cannot be tested faithfully, with decline_code, a one-sentence decline_reason, and a testable suggestion:
- options_or_futures: option legs, strikes, premiums, straddles, futures rollovers (suggest the same signal on the underlying).
- multiple_instruments: pairs, baskets, scanning many stocks, or a signal from one symbol traded on another.
- multiple_timeframes: rules mixing candle sizes (e.g. daily trend filter with 5-min entries).
- fundamentals: P/E, earnings, results, news, sentiment.
- unsupported_data: open interest, order book, India VIX as a filter on another symbol.
- unsupported_indicator: an indicator that is not in the list and cannot be built from it.
- unsupported_logic: needs nested AND/OR such as "(A and B) or C".
- look_ahead: uses future information.
- too_vague: no entry rule can be identified. not_a_strategy: not a trading request.
"""

_EXAMPLE = {
    "outcome": "strategy",
    "strategy": {
        "name": "NIFTY 9/21 EMA crossover",
        "symbol": "NIFTY",
        "market": "india",
        "side": "long",
        "interval": "5m",
        "style": "intraday",
        "period": {"lookback_days": 365},
        "entry": {
            "match": "all",
            "rules": [
                {
                    "type": "compare",
                    "left": {"kind": "indicator", "name": "ema", "period": 9},
                    "op": "crosses_above",
                    "right": {"kind": "indicator", "name": "ema", "period": 21},
                }
            ],
        },
        "exit": {
            "match": "any",
            "rules": [
                {
                    "type": "compare",
                    "left": {"kind": "indicator", "name": "rsi", "period": 14},
                    "op": ">",
                    "right": {"kind": "value", "value": 70},
                }
            ],
        },
        "risk": {"stop_loss": {"unit": "percent", "value": 2}, "take_profit": {"unit": "percent", "value": 4}},
    },
    "assumptions": [
        {"field": "side", "value": "long", "reason": "A cross above suggests buying."},
        {"field": "style", "value": "intraday", "reason": "5-minute candles with no mention of holding overnight."},
    ],
}


def _openai_instructions() -> str:
    schema = json.dumps(ParseResult.model_json_schema(), separators=(",", ":"))
    return (
        SYSTEM_PROMPT
        + "\nReply with ONE JSON object and nothing else. It must match this JSON Schema:\n"
        + schema
        + "\n\nExample reply for 'Nifty 9 EMA crosses above 21 EMA on 5-min, 2% SL, 4% target or exit when RSI above 70, last 1 year':\n"
        + json.dumps(_EXAMPLE)
    )


# ---------------------------------------------------------------- semantic checks


def semantic_errors(s: Strategy) -> list[str]:
    """Problems the JSON schema cannot catch but the engine would reject."""
    errors = []
    rule_sets = [s.entry] + ([s.exit] if s.exit else [])
    for rs in rule_sets:
        for r in rs.rules:
            if isinstance(r, TimeWindowRule) and s.interval == "1d":
                errors.append("time_window rules need an intraday interval")
            if isinstance(r, CompareRule):
                for op in (r.left, r.right):
                    if isinstance(op, IndicatorOperand):
                        spec = indicators.REGISTRY[op.name]
                        if op.output is not None and op.output not in spec.outputs:
                            errors.append(f"{op.name} output must be one of {list(spec.outputs)}, got '{op.output}'")
                        if spec.intraday_only and s.interval == "1d":
                            errors.append(f"{op.name} needs an intraday interval")
                        bar = indicators.INTERVAL_MINUTES[s.interval]
                        if op.name in ("orb_high", "orb_low") and bar and (op.minutes or 15) % bar:
                            errors.append(f"{op.name} minutes must be a multiple of the {bar}-minute candle")
    return errors


# ---------------------------------------------------------------- providers


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start : end + 1])


def _anthropic_chat(cfg: LLMConfig, messages: list[dict]) -> str:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise LLMNotConfigured("Install the Claude client: pip install 'plainquant[anthropic]'") from e
    client = anthropic.Anthropic(api_key=cfg.api_key) if cfg.api_key else anthropic.Anthropic()
    # The strategy schema is beyond the size limits of constrained decoding, so the schema goes in the
    # prompt and the reply is validated (and repaired) locally, the same way as other providers.
    kwargs = {"model": cfg.model, "max_tokens": 16000, "system": _openai_instructions(), "messages": messages}
    if cfg.model in _FALLBACK_MODELS:
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        resp = client.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    if resp.stop_reason == "max_tokens":
        raise LLMError("The model ran out of output tokens.")
    return "".join(b.text for b in resp.content if b.type == "text")


def _openai_chat(cfg: LLMConfig, messages: list[dict]) -> str:
    try:
        import openai
    except ImportError as e:  # pragma: no cover
        raise LLMNotConfigured("Install the OpenAI-compatible client: pip install 'plainquant[openai]'") from e
    client = openai.OpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "not-needed")
    full = [{"role": "system", "content": _openai_instructions()}] + messages
    try:
        resp = client.chat.completions.create(model=cfg.model, messages=full, response_format={"type": "json_object"})
    except openai.BadRequestError:
        resp = client.chat.completions.create(model=cfg.model, messages=full)  # server without JSON mode
    return resp.choices[0].message.content or ""


def parse_strategy(prompt: str, cfg: LLMConfig | None = None, max_repairs: int = 2, _chat=None) -> ParseResult:
    """Ask the configured model to turn `prompt` into a strategy. Validates and repairs."""
    cfg = cfg or config_from_env()
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Describe a strategy first.")
    messages: list[dict] = [{"role": "user", "content": prompt}]
    last_error = ""
    for _ in range(max_repairs + 1):
        chat = _chat or (_anthropic_chat if cfg.kind == "anthropic" else _openai_chat)
        reply_text = chat(cfg, messages)
        try:
            result = ParseResult.model_validate(_extract_json(reply_text))
            if result.outcome == "strategy":
                errs = semantic_errors(result.strategy)
                if errs:
                    raise ValueError("; ".join(errs))
            return result
        except (ValidationError, ValueError) as e:
            last_error = str(e)[:2000]
            messages.append({"role": "assistant", "content": reply_text or "(empty reply)"})
            messages.append({"role": "user", "content": f"That reply is invalid: {last_error}\nSend the corrected JSON only."})
    raise LLMError(f"The model could not produce a valid strategy: {last_error}")
