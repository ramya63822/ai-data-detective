
"""Optional language-model layer: Gemini, OpenAI, or local Ollama.

The model maps questions to validated analysis plans and interprets verified
evidence. Numeric results must be computed by the deterministic analytics code.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from .fmt import extract_numbers

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

T = TypeVar("T", bound=BaseModel)


def _setting(name: str, default: str = "") -> str:
    """Read configuration from environment/.env or Streamlit secrets."""
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()

    try:
        import streamlit as st

        value = st.secrets.get(name, default)
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass

    return default


def _cfg():
    return {
        "provider": _setting("AI_PROVIDER", "gemini").lower(),
        "gemini_model": _setting(
            "GEMINI_MODEL", "gemini-flash-latest"
        ),
        "gemini_key": _setting("GEMINI_API_KEY"),
        "openai_model": _setting("OPENAI_MODEL", "gpt-6-luna"),
        "openai_key": _setting("OPENAI_API_KEY"),
        "ollama_model": _setting("OLLAMA_MODEL", "qwen3:4b"),
    }


def provider_name() -> str:
    """Return the selected LLM provider."""
    return _cfg()["provider"]


def available() -> bool:
    """Check whether the selected provider's SDK and configuration exist."""
    c = _cfg()

    if c["provider"] == "gemini":
        if not c["gemini_key"]:
            return False
        try:
            from google import genai  # noqa: F401
            return True
        except ImportError:
            return False

    if c["provider"] == "openai":
        if not c["openai_key"]:
            return False
        try:
            from openai import OpenAI  # noqa: F401
            return True
        except ImportError:
            return False

    if c["provider"] == "ollama":
        try:
            import ollama  # noqa: F401
            return True
        except ImportError:
            return False

    return False


def _openai_client(api_key: str):
    """Create an OpenAI client with bounded retries and timeout."""
    from openai import OpenAI

    return OpenAI(api_key=api_key, timeout=35.0, max_retries=1)


def _text(
    prompt: str,
    system: str,
    max_tokens: int,
) -> Optional[str]:
    """Generate text using the configured provider."""
    c = _cfg()

    try:
        if c["provider"] == "gemini" and c["gemini_key"]:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=c["gemini_key"],
                http_options={"timeout": 35000},
            )

            response = client.models.generate_content(
                model=c["gemini_model"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.2,
                ),
            )
            return (response.text or "").strip() or None

        if c["provider"] == "openai" and c["openai_key"]:
            response = _openai_client(
                c["openai_key"]
            ).responses.create(
                model=c["openai_model"],
                instructions=system,
                input=prompt,
                max_output_tokens=max_tokens,
            )
            return (response.output_text or "").strip() or None

        if c["provider"] == "ollama":
            import ollama

            response = ollama.chat(
                model=c["ollama_model"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "num_ctx": 2048,
                    "num_predict": max_tokens,
                },
                think=False,
            )
            return (
                response["message"]["content"].strip() or None
            )

    except Exception:
        # Preserve deterministic app functionality if the LLM fails.
        return None

    return None


def plan(prompt: str, schema: Type[T]) -> Optional[T]:
    """Generate a structured plan and validate it against the Pydantic schema."""
    c = _cfg()

    try:
        if c["provider"] == "gemini" and c["gemini_key"]:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=c["gemini_key"],
                http_options={"timeout": 35000},
            )

            response = client.models.generate_content(
                model=c["gemini_model"],
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                    max_output_tokens=400,
                ),
            )

            if not response.text:
                return None

            return schema.model_validate_json(response.text)

        if c["provider"] == "openai" and c["openai_key"]:
            response = _openai_client(
                c["openai_key"]
            ).responses.parse(
                model=c["openai_model"],
                input=prompt,
                text_format=schema,
            )
            return response.output_parsed

        if c["provider"] == "ollama":
            import ollama

            response = ollama.chat(
                model=c["ollama_model"],
                messages=[{"role": "user", "content": prompt}],
                format=schema.model_json_schema(),
                options={
                    "num_ctx": 2048,
                    "num_predict": 400,
                },
                think=False,
            )
            return schema.model_validate_json(
                response["message"]["content"]
            )

    except Exception:
        return None

    return None


_SYSTEM = (
    "You are a senior business analyst writing for executives. "
    "You are given VERIFIED EVIDENCE computed by software. "
    "Write a brief interpretation: what the evidence suggests and "
    "how confident one can be. Use only numeric values directly "
    "supported by the evidence. Never introduce new numbers, "
    "including counts, causes, or business context. Never claim "
    "causation from association. Do not give recommendations. "
    "No headings or bullet lists. Write 2-4 sentences."
)


def grounded(
    text: str,
    evidence: str,
    rel_tol: float = 0.005,
) -> bool:
    """Reject interpretations containing unsupported numeric values."""
    allowed = extract_numbers(evidence)

    for number in extract_numbers(text):
        if not any(
            abs(number - value)
            <= max(rel_tol * abs(value), 0.051)
            for value in allowed
        ):
            return False

    return True


def interpret(
    question: str,
    evidence: str,
) -> Optional[str]:
    """Interpret verified evidence, rejecting unsupported numeric claims."""
    if not available():
        return None

    prompt = (
        f"Question: {question}\n\n"
        f"VERIFIED EVIDENCE:\n{evidence}\n\n"
        "Interpretation:"
    )

    output = _text(prompt, _SYSTEM, 260)

    if not output:
        return None

    output = re.sub(
        r"^#+\s.*$",
        "",
        output,
        flags=re.M,
    ).replace("**", "").strip()

    if len(output) > 900 or not grounded(output, evidence):
        return None

    return output


def test_gemini_connection() -> tuple[bool, str]:
    """Send a small live request to verify Gemini connectivity."""
    c = _cfg()

    if c["provider"] != "gemini":
        return False, "Set AI_PROVIDER=gemini to test Gemini."

    if not c["gemini_key"]:
        return False, (
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    try:
        from google import genai

        client = genai.Client(
            api_key=c["gemini_key"],
            http_options={"timeout": 20000},
        )

        response = client.models.generate_content(
            model=c["gemini_model"],
            contents="Reply with exactly: OK",
        )

        if not (response.text or "").strip():
            return False, "Gemini returned no text."

        return True, (
            f"Gemini connection successful. "
            f"Model {c['gemini_model']} responded."
        )

    except Exception as exc:
        error_name = type(exc).__name__

        if error_name in {
            "AuthenticationError",
            "APIKeyError",
            "Unauthorized",
        }:
            message = (
                "Gemini rejected the API key. "
                "Check that your key is valid and active."
            )
        elif error_name in {
            "ClientError",
            "PermissionDenied",
            "Forbidden",
        }:
            message = (
                "Gemini rejected the request. Check model access, "
                "API key permissions, and quota."
            )
        elif error_name in {
            "ServerError",
            "APIError",
        }:
            message = (
                "Gemini API request failed. Check model availability, "
                "service status, and quota."
            )
        else:
            message = (
                f"Gemini request failed ({error_name}). "
                "Check your API key, model access, quota, and internet."
            )

        return False, message


def test_openai_connection() -> tuple[bool, str]:
    """Make a small live request to verify OpenAI connectivity."""
    c = _cfg()

    if c["provider"] != "openai":
        return False, (
            "Set AI_PROVIDER=openai to test the OpenAI connection."
        )

    if not c["openai_key"]:
        return False, (
            "OPENAI_API_KEY is missing. Configure it in .env "
            "or Streamlit secrets."
        )

    try:
        from openai import OpenAI
    except ImportError:
        return False, (
            "The OpenAI SDK is missing. Run: "
            "python -m pip install openai"
        )

    try:
        client = OpenAI(
            api_key=c["openai_key"],
            timeout=20.0,
            max_retries=0,
        )

        response = client.responses.create(
            model=c["openai_model"],
            input="Reply with exactly: OK",
            max_output_tokens=8,
        )

        if not (getattr(response, "output_text", "") or "").strip():
            return False, (
                "The API request returned no text. Check model support."
            )

        return True, (
            f"OpenAI connection successful. "
            f"Model {c['openai_model']} responded."
        )

    except Exception as exc:
        error_name = type(exc).__name__

        if error_name == "AuthenticationError":
            message = (
                "OpenAI rejected the API key. Check OPENAI_API_KEY."
            )
        elif error_name in {
            "PermissionDeniedError",
            "NotFoundError",
        }:
            message = (
                f"Check access to OpenAI model {c['openai_model']}."
            )
        elif error_name == "RateLimitError":
            message = (
                "OpenAI rate limit or quota reached. "
                "Check project limits and billing."
            )
        elif error_name in {
            "APIConnectionError",
            "APITimeoutError",
            "TimeoutError",
        }:
            message = (
                "Could not reach OpenAI. Check your internet connection."
            )
        else:
            message = (
                f"OpenAI request failed ({error_name}). "
                "Check API key, model access, quota, and connectivity."
            )

        return False, message


def status_label() -> str:
    """Return a user-friendly status for the selected provider."""
    c = _cfg()

    if c["provider"] == "gemini":
        if c["gemini_key"]:
            return f"Gemini configured ({c['gemini_model']})"
        return "Gemini not configured (add GEMINI_API_KEY)"

    if c["provider"] == "openai":
        if c["openai_key"]:
            return f"OpenAI configured ({c['openai_model']})"
        return "OpenAI not configured (add OPENAI_API_KEY)"

    if c["provider"] == "ollama":
        return (
            "Ollama LLM: enabled"
            if available()
            else "Ollama LLM: unavailable"
        )

    return "LLM: disabled (analysis fully available)"
