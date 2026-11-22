import json

import pytest

from plainquant import llm
from plainquant.dsl import ParseResult, Strategy
from plainquant.llm import LLMConfig, LLMNotConfigured, config_from_env, parse_strategy

CFG = LLMConfig(provider="ollama", kind="openai", model="test", base_url="http://localhost:11434/v1")


def good_reply():
    return json.dumps(llm._EXAMPLE)


def test_parses_valid_reply():
    result = parse_strategy("nifty ema cross", CFG, _chat=lambda cfg, m: good_reply())
    assert result.outcome == "strategy"
    assert result.strategy.symbol == "NIFTY"


def test_accepts_fenced_json():
    result = parse_strategy("x", CFG, _chat=lambda cfg, m: f"Sure!\n```json\n{good_reply()}\n```")
    assert result.strategy.interval == "5m"


def test_repairs_invalid_reply():
    calls = []

    def chat(cfg, messages):
        calls.append(list(messages))
        if len(calls) == 1:
            bad = json.loads(good_reply())
            bad["strategy"]["entry"]["rules"][0]["op"] = "goes_above"
            return json.dumps(bad)
        return good_reply()

    result = parse_strategy("x", CFG, _chat=chat)
    assert result.outcome == "strategy"
    assert len(calls) == 2
    assert "invalid" in calls[1][-1]["content"]


def test_semantic_errors_trigger_repair():
    bad = json.loads(good_reply())
    bad["strategy"]["interval"] = "1d"
    bad["strategy"]["style"] = "positional"
    bad["strategy"]["entry"]["rules"].append({"type": "time_window", "start": "09:30", "end": "10:00"})
    replies = iter([json.dumps(bad), good_reply()])
    result = parse_strategy("x", CFG, _chat=lambda c, m: next(replies))
    assert result.strategy.interval == "5m"


def test_gives_up_after_repairs():
    with pytest.raises(llm.LLMError):
        parse_strategy("x", CFG, _chat=lambda c, m: "not json at all")


def test_decline_round_trip():
    reply = {"outcome": "decline", "decline_code": "options_or_futures", "decline_reason": "Needs option prices.", "suggestion": "Test the signal on NIFTY."}
    result = parse_strategy("nifty straddle", CFG, _chat=lambda c, m: json.dumps(reply))
    assert result.outcome == "decline"
    assert result.decline_code == "options_or_futures"


def test_schema_is_not_recursive():
    schema = json.dumps(ParseResult.model_json_schema())
    defs = ParseResult.model_json_schema()["$defs"]
    # a recursive schema would reference a definition from inside itself
    for name, body in defs.items():
        assert f'"#/$defs/{name}"' not in json.dumps(body), name
    assert len(schema) < 20000


def test_config_from_env(monkeypatch):
    for k in ("PLAINQUANT_PROVIDER", "PLAINQUANT_MODEL", "PLAINQUANT_BASE_URL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PLAINQUANT_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(LLMNotConfigured):
        config_from_env()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert config_from_env().model == "claude-opus-5-5"
    monkeypatch.setenv("PLAINQUANT_PROVIDER", "ollama")
    with pytest.raises(LLMNotConfigured):
        config_from_env()  # ollama needs a model name
    monkeypatch.setenv("PLAINQUANT_MODEL", "llama")
    cfg = config_from_env()
    assert cfg.kind == "openai" and cfg.base_url.startswith("http://localhost:11434")


def test_example_strategy_in_prompt_is_valid():
    Strategy.model_validate(llm._EXAMPLE["strategy"])
    ParseResult.model_validate(llm._EXAMPLE)
