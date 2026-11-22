"""Local web app: `plainquant serve`.

Runs on your machine. Your model key stays in your environment and prompts go only to the
model endpoint you configured.
"""

from __future__ import annotations

import os
from importlib import resources

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ValidationError

from . import examples, settings
from .data import DataProvider, YahooProvider
from .dsl import Strategy
from .engine import run
from .explain import explain
from .llm import (
    LLMConfig,
    LLMError,
    LLMNotConfigured,
    config_for,
    config_from_env,
    config_source,
    list_models,
    parse_strategy,
    providers,
    semantic_errors,
)
from .report import to_dict

# Files the page loads. Bundled so the app works offline and makes no third-party requests.
ASSETS = {"Doto.woff2": "font/woff2", "GeistMono.woff2": "font/woff2", "hero.webp": "image/webp"}


class PromptIn(BaseModel):
    prompt: str


class StrategyIn(BaseModel):
    strategy: dict


class ModelChoice(BaseModel):
    provider: str
    model: str = ""
    base_url: str | None = None
    api_key: str | None = None


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def local_only(request: Request) -> None:
    """Model settings can only be read or changed by a browser on this computer, on this app's page.

    Blocks other machines (when served on 0.0.0.0), DNS-rebinding pages, and other websites
    posting to localhost.
    """
    host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
    client = request.client.host if request.client else ""
    origin = request.headers.get("origin")
    same_origin = origin is None or origin.split("://", 1)[-1] == request.headers.get("host")
    if host not in LOOPBACK or client not in LOOPBACK or not same_origin:
        raise HTTPException(403, detail=["Model settings can only be changed from this computer."])


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

    def settings_view() -> dict:
        try:
            cfg = llm_config()
        except LLMNotConfigured:
            cfg = None
        key = cfg.api_key if cfg else None
        return {
            "connected": cfg is not None,
            "source": "app" if llm is None and config_source() == "app" else ("environment" if cfg else None),
            "provider": cfg.provider if cfg else None,
            "model": cfg.model if cfg else None,
            "base_url": cfg.base_url if cfg else None,
            "key_hint": f"…{key[-4:]}" if key and len(key) > 8 else ("set" if key else None),
            "path": str(settings.path()),
            "env_locked": llm is None and bool(os.environ.get("PLAINQUANT_PROVIDER")),
            "providers": providers(),
        }

    def chosen(body: ModelChoice) -> LLMConfig:
        cfg = config_for(body.provider, body.model, body.base_url, body.api_key)
        saved = settings.load()
        # Reuse the saved key when the user only changes the model on the same endpoint.
        if not cfg.api_key and saved and saved["provider"] == cfg.provider:
            same_url = (saved["base_url"] or config_for(saved["provider"]).base_url) == cfg.base_url
            if same_url:
                cfg.api_key = saved["api_key"]
        return cfg

    def fetch_models(cfg: LLMConfig) -> list[str]:
        try:
            return list_models(cfg)
        except LLMNotConfigured as e:
            raise HTTPException(400, detail=[str(e)]) from e
        except Exception as e:  # bad key, wrong URL, server not running
            raise HTTPException(400, detail=[_friendly(e, cfg)]) from e

    @app.get("/api/settings", dependencies=[Depends(local_only)])
    def get_settings():
        return settings_view()

    @app.post("/api/settings/models", dependencies=[Depends(local_only)])
    def settings_models(body: ModelChoice):
        cfg = chosen(body)
        return {"models": fetch_models(cfg)}

    @app.post("/api/settings", dependencies=[Depends(local_only)])
    def save_settings(body: ModelChoice):
        cfg = chosen(body)
        if not cfg.model:
            raise HTTPException(400, detail=["Pick a model."])
        models = fetch_models(cfg)
        if models and cfg.model not in models:
            raise HTTPException(400, detail=[f"{cfg.model} isn't offered by this provider. Pick one from the list."])
        custom = cfg.base_url if cfg.base_url != config_for(cfg.provider).base_url else None
        settings.save(cfg.provider, cfg.model, custom, cfg.api_key)
        return settings_view()

    @app.delete("/api/settings", dependencies=[Depends(local_only)])
    def forget_settings():
        settings.clear()
        return settings_view()

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


def _friendly(e: Exception, cfg: LLMConfig) -> str:
    text = str(e)
    status = getattr(e, "status_code", None)
    if status in (401, 403):
        return "The key was rejected. Check that you pasted the whole key."
    if "Connection" in type(e).__name__ or "connect" in text.lower():
        where = cfg.base_url or "the provider"
        return f"Couldn't reach {where}. If it runs on your computer, make sure it's started."
    return f"The provider said: {text[:300]}"


def _errors(e: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()][:20]
