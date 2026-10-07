"""Configuration and connection-check tests for the optional OpenAI layer."""
from __future__ import annotations

import sys
import types

from detective import llm


def _cfg(key: str = "test-key", model: str = "gpt-6-luna", provider: str = "openai"):
    return {
        "provider": provider,
        "openai_model": model,
        "openai_key": key,
        "ollama_model": "qwen3:4b",
    }


def test_openai_connection_requires_key(monkeypatch):
    monkeypatch.setattr(llm, "_cfg", lambda: _cfg(key=""))
    ok, message = llm.test_openai_connection()
    assert not ok
    assert "OPENAI_API_KEY is missing" in message


def test_openai_connection_success_uses_configured_model(monkeypatch):
    calls = {}

    class FakeResponse:
        output_text = "OK"

    class FakeResponses:
        def create(self, **kwargs):
            calls.update(kwargs)
            return FakeResponse()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == "test-key"
            self.responses = FakeResponses()

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setattr(llm, "_cfg", lambda: _cfg(model="gpt-6-luna"))

    ok, message = llm.test_openai_connection()
    assert ok
    assert "connection successful" in message
    assert calls["model"] == "gpt-6-luna"
    assert calls["max_output_tokens"] == 8


def test_openai_connection_failure_does_not_show_exception_text(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = self

        def create(self, **kwargs):
            raise RuntimeError("sensitive test detail")

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setattr(llm, "_cfg", lambda: _cfg(key="secret-test-key"))

    ok, message = llm.test_openai_connection()
    assert not ok
    assert "RuntimeError" in message
    assert "sensitive test detail" not in message
    assert "secret-test-key" not in message


def test_streamlit_secrets_can_configure_openai(monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.secrets = {
        "OPENAI_API_KEY": "cloud-test-key",
        "AI_PROVIDER": "openai",
        "OPENAI_MODEL": "gpt-6-luna",
    }
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert llm._setting("OPENAI_API_KEY") == "cloud-test-key"
    cfg = llm._cfg()
    assert cfg["provider"] == "openai"
    assert cfg["openai_model"] == "gpt-6-luna"


def test_status_reports_configured_provider_without_claiming_a_successful_request(monkeypatch):
    monkeypatch.setattr(llm, "_cfg", lambda: _cfg())
    label = llm.status_label()
    assert "OpenAI configured" in label
    assert "successful" not in label
