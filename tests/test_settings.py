"""Model setup from the web app: saved settings, key handling and local-only access."""

import os
import stat

import pytest
from fastapi.testclient import TestClient

from plainquant import server, settings
from plainquant.llm import config_from_env


@pytest.fixture
def client(monkeypatch):
    for k in ("PLAINQUANT_PROVIDER", "PLAINQUANT_MODEL", "PLAINQUANT_BASE_URL", "PLAINQUANT_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    seen = []

    def fake_models(cfg):
        seen.append(cfg)
        if cfg.api_key == "bad":
            raise RuntimeError("401")
        return ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]

    monkeypatch.setattr(server, "list_models", fake_models)
    c = TestClient(server.create_app(provider=object()), base_url="http://127.0.0.1:8000", client=("127.0.0.1", 5000))
    c.seen = seen
    return c


def test_save_from_the_app_and_use_it_everywhere(client):
    r = client.post("/api/settings", json={"provider": "groq", "model": "openai/gpt-oss-120b", "api_key": "gsk_secret_1234"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["connected"] and body["source"] == "app" and body["key_hint"] == "…1234"
    assert "gsk_secret_1234" not in r.text and "gsk_secret_1234" not in client.get("/api/settings").text
    if os.name == "posix":
        assert stat.S_IMODE(settings.path().stat().st_mode) == 0o600
    cfg = config_from_env()
    assert (cfg.provider, cfg.model, cfg.api_key) == ("groq", "openai/gpt-oss-120b", "gsk_secret_1234")
    assert cfg.base_url == "https://api.groq.com/openai/v1"


def test_changing_only_the_model_keeps_the_saved_key(client):
    client.post("/api/settings", json={"provider": "groq", "model": "openai/gpt-oss-120b", "api_key": "gsk_secret_1234"})
    r = client.post("/api/settings", json={"provider": "groq", "model": "llama-3.3-70b-versatile"})
    assert r.status_code == 200 and settings.load()["api_key"] == "gsk_secret_1234"


def test_saved_key_is_not_sent_to_another_server(client):
    client.post("/api/settings", json={"provider": "groq", "model": "openai/gpt-oss-120b", "api_key": "gsk_secret_1234"})
    client.post("/api/settings/models", json={"provider": "groq", "base_url": "http://evil.example/v1"})
    assert client.seen[-1].api_key is None
    assert config_from_env(base_url="http://evil.example/v1").api_key is None


def test_bad_key_and_unknown_model_are_not_saved(client):
    r = client.post("/api/settings", json={"provider": "groq", "model": "openai/gpt-oss-120b", "api_key": "bad"})
    assert r.status_code == 400 and settings.load() is None
    r = client.post("/api/settings", json={"provider": "groq", "model": "gpt-9", "api_key": "gsk_ok_12345678"})
    assert r.status_code == 400 and "isn't offered" in r.text and settings.load() is None


def test_forget(client):
    client.post("/api/settings", json={"provider": "ollama", "model": "openai/gpt-oss-120b"})
    assert client.delete("/api/settings").json()["connected"] is False
    assert settings.load() is None


def test_shell_provider_wins_over_saved_settings(client, monkeypatch):
    client.post("/api/settings", json={"provider": "ollama", "model": "openai/gpt-oss-120b"})
    monkeypatch.setenv("PLAINQUANT_PROVIDER", "groq")
    assert config_from_env().provider == "groq"
    assert client.get("/api/settings").json()["env_locked"] is True


@pytest.mark.parametrize(
    "base_url, client_addr, headers",
    [
        ("http://evil.example", ("127.0.0.1", 1), {}),  # DNS rebinding
        ("http://127.0.0.1:8000", ("192.168.1.9", 1), {}),  # another machine on the network
        ("http://127.0.0.1:8000", ("127.0.0.1", 1), {"Origin": "https://evil.example"}),  # another website
    ],
)
def test_settings_are_local_only(client, base_url, client_addr, headers):
    c = TestClient(server.create_app(provider=object()), base_url=base_url, client=client_addr)
    assert c.get("/api/settings", headers=headers).status_code == 403
    r = c.post("/api/settings", json={"provider": "ollama", "model": "m", "base_url": "http://evil.example/v1"}, headers=headers)
    assert r.status_code == 403 and settings.load() is None
