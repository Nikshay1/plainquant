"""Command line interface: `plainquant --help`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, examples, indicators
from .data import CSVProvider, YahooProvider
from .dsl import ParseResult, Strategy
from .engine import run
from .explain import explain
from .llm import LLMError, LLMNotConfigured, config_from_env, parse_strategy
from .report import to_dict, to_html, to_text


def _provider(args):
    return CSVProvider(args.csv, timezone=args.csv_tz) if args.csv else YahooProvider()


def _print_parse(result: ParseResult) -> None:
    if result.outcome == "decline":
        print(f"Can't test this as written ({result.decline_code}): {result.decline_reason}")
        if result.suggestion:
            print(f"Try instead: {result.suggestion}")
        return
    print("Here is how I read your strategy:\n")
    for line in explain(result.strategy):
        print(f"  - {line}")
    if result.assumptions:
        print("\nFilled in for you:")
        for a in result.assumptions:
            print(f"  - {a.field} = {a.value} ({a.reason})")


def _llm(args):
    return config_from_env(args.provider, args.model, args.base_url)


def _report(result, args) -> None:
    print(to_text(result))
    if args.json:
        Path(args.json).write_text(json.dumps(to_dict(result), indent=2, default=str))
        print(f"Saved {args.json}", file=sys.stderr)
    if args.html:
        Path(args.html).write_text(to_html(result))
        print(f"Saved {args.html}", file=sys.stderr)


def cmd_run(args) -> int:
    cfg = _llm(args)
    print(f"Reading your strategy with {cfg.label()} ...\n", file=sys.stderr)
    parsed = parse_strategy(" ".join(args.prompt), cfg)
    _print_parse(parsed)
    if parsed.outcome == "decline":
        return 2
    if args.save:
        Path(args.save).write_text(parsed.strategy.model_dump_json(indent=2, exclude_none=True))
        print(f"\nSaved the strategy to {args.save}. Edit it and run: plainquant backtest {args.save}")
    if not args.yes:
        if not sys.stdin.isatty():
            print("\nAdd --yes to run without confirming.")
            return 0
        if input("\nRun the backtest? [Y/n] ").strip().lower() not in ("", "y", "yes"):
            return 0
    print()
    _report(run(parsed.strategy, _provider(args)), args)
    return 0


def cmd_parse(args) -> int:
    parsed = parse_strategy(" ".join(args.prompt), _llm(args))
    if args.output and parsed.outcome == "strategy":
        Path(args.output).write_text(parsed.strategy.model_dump_json(indent=2, exclude_none=True))
        _print_parse(parsed)
        print(f"\nSaved {args.output}")
    else:
        print(parsed.model_dump_json(indent=2, exclude_none=True))
    return 0 if parsed.outcome == "strategy" else 2


def cmd_backtest(args) -> int:
    if args.example:
        strategy = examples.load(args.example)
    elif args.file:
        strategy = Strategy.model_validate_json(Path(args.file).read_text())
    else:
        print("Give a strategy file or --example NAME", file=sys.stderr)
        return 1
    for line in explain(strategy):
        print(f"  - {line}")
    print()
    _report(run(strategy, _provider(args)), args)
    return 0


def cmd_examples(args) -> int:
    for name in examples.names():
        print(f"{name:<24} {examples.load(name).name}")
    return 0


def cmd_indicators(args) -> int:
    print(indicators.describe())
    return 0


def cmd_schema(args) -> int:
    print(json.dumps(Strategy.model_json_schema(), indent=2))
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Install the web app extras: pip install 'plainquant[server]'", file=sys.stderr)
        return 1
    from .server import create_app

    print(f"plainquant web app on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="plainquant", description="Backtest trading strategies written in plain English.")
    p.add_argument("--version", action="version", version=f"plainquant {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def llm_opts(sp):
        sp.add_argument("--provider", help="anthropic, openai, ollama, lmstudio, openrouter, groq")
        sp.add_argument("--model", help="model name for your provider")
        sp.add_argument("--base-url", help="OpenAI-compatible endpoint URL")

    def data_opts(sp):
        sp.add_argument("--csv", help="use candles from this CSV instead of Yahoo Finance")
        sp.add_argument("--csv-tz", help="time zone of naive CSV timestamps, e.g. Asia/Kolkata")
        sp.add_argument("--json", help="also save the full report as JSON")
        sp.add_argument("--html", help="also save a shareable HTML report")

    sp = sub.add_parser("run", help="describe a strategy, check it, backtest it")
    sp.add_argument("prompt", nargs="+")
    sp.add_argument("-y", "--yes", action="store_true", help="run without asking to confirm the rules")
    sp.add_argument("--save", help="save the parsed strategy JSON here")
    llm_opts(sp)
    data_opts(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("parse", help="turn a description into strategy JSON without running it")
    sp.add_argument("prompt", nargs="+")
    sp.add_argument("-o", "--output", help="write the strategy JSON to this file")
    llm_opts(sp)
    sp.set_defaults(func=cmd_parse)

    sp = sub.add_parser("backtest", help="backtest a strategy JSON file (no model needed)")
    sp.add_argument("file", nargs="?")
    sp.add_argument("--example", help="run a bundled example, see `plainquant examples`")
    data_opts(sp)
    sp.set_defaults(func=cmd_backtest)

    sub.add_parser("examples", help="list bundled example strategies").set_defaults(func=cmd_examples)
    sub.add_parser("indicators", help="list indicators").set_defaults(func=cmd_indicators)
    sub.add_parser("schema", help="print the strategy JSON Schema").set_defaults(func=cmd_schema)

    sp = sub.add_parser("serve", help="start the local web app")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LLMNotConfigured as e:
        print(e, file=sys.stderr)
        return 1
    except (LLMError, ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # provider SDK errors: bad key, unreachable endpoint, rate limits
        if os.environ.get("PLAINQUANT_DEBUG"):
            raise
        print(f"Error: {type(e).__name__}: {e}\n(set PLAINQUANT_DEBUG=1 for the full traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
