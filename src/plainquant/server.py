"""Local web app: `plainquant serve`.

Runs on your machine. Your model key stays in your environment and prompts go only to the
model endpoint you configured.
"""

from __future__ import annotations

from importlib import resources

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ValidationError

from . import examples
from .data import DataProvider, YahooProvider
from .dsl import Strategy
from .engine import run
from .explain import explain
from .llm import LLMConfig, LLMError, LLMNotConfigured, config_from_env, parse_strategy, semantic_errors
from .report import to_dict

# Files the page loads. Bundled so the app works offline and makes no third-party requests.
ASSETS = {"Doto.woff2": "font/woff2", "GeistMono.woff2": "font/woff2", "hero.webp": "image/webp"}


class PromptIn(BaseModel):
    prompt: str


class StrategyIn(BaseModel):
    strategy: dict


def create_app(provider: DataProvider | None = None, llm: LLMConfig | None = None) -> FastAPI:
    app = FastAPI(title="plainquant", docs_url="/api/docs")
    provider = provider or YahooProvider()
    page = (resources.files("plainquant.web") / "index.html").read_text()

    def llm_config() -> LLMConfig:
        return llm or config_from_env()

    def validate(raw: dict) -> Strategy:
        try:
            s = Strategy.model_validate(raw)
        except ValidationError as e:
            raise HTTPException(400, detail=_errors(e)) from e
        errs = semantic_errors(s)
        if errs:
            raise HTTPException(400, detail=errs)
        return s

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page

    web = resources.files("plainquant.web")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in ASSETS:
            raise HTTPException(404)
        headers = {"Cache-Control": "max-age=86400"}
        return Response((web / "assets" / name).read_bytes(), media_type=ASSETS[name], headers=headers)

    @app.get("/api/status")
    def status():
        try:
            return {"llm": llm_config().label(), "help": None}
        except LLMNotConfigured as e:
            return {"llm": None, "help": str(e)}

    @app.get("/api/examples")
    def list_examples():
        out = []
        for name in examples.names():
            s = examples.load(name)
            out.append({"id": name, "name": s.name, "strategy": s.model_dump(mode="json", exclude_none=True), "rules": explain(s)})
        return out

    @app.post("/api/parse")
    def parse(body: PromptIn):
        try:
            result = parse_strategy(body.prompt, llm_config())
        except LLMNotConfigured as e:
            raise HTTPException(503, detail=[str(e)]) from e
        except (LLMError, ValueError) as e:
            raise HTTPException(422, detail=[str(e)]) from e
        except Exception as e:  # provider errors: bad key, network, rate limits
            raise HTTPException(502, detail=[f"Model call failed: {e}"]) from e
        out = result.model_dump(mode="json", exclude_none=True)
        if result.strategy is not None:
            out["rules"] = explain(result.strategy)
        return out

    @app.post("/api/explain")
    def explain_strategy(body: StrategyIn):
        return {"rules": explain(validate(body.strategy))}

    @app.post("/api/backtest")
    def backtest(body: StrategyIn):
        s = validate(body.strategy)
        try:
            return to_dict(run(s, provider))
        except ValueError as e:
            raise HTTPException(422, detail=[str(e)]) from e

    return app


def _errors(e: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()][:20]
