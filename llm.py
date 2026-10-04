from __future__ import annotations

import os
from typing import Optional, Type, TypeVar

import ollama
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

PROVIDER = os.getenv("AI_PROVIDER", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

_openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    _openai_client = OpenAI(api_key=OPENAI_API_KEY)


def provider_name() -> str:
    if PROVIDER == "openai" and _openai_client:
        return f"OpenAI · {OPENAI_MODEL}"
    if PROVIDER == "ollama":
        return f"Ollama · {OLLAMA_MODEL}"
    if PROVIDER == "openai" and not _openai_client:
        return "OpenAI key missing"
    return PROVIDER


def _openai_text(prompt: str, max_output_tokens: int, system_instruction: Optional[str]) -> str:
    if not _openai_client:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    instructions = system_instruction or "You are a precise business data analyst."
    response = _openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    if not response.output_text:
        raise RuntimeError("OpenAI returned an empty response.")
    return response.output_text.strip()


def _ollama_text(prompt: str, max_output_tokens: int, system_instruction: Optional[str]) -> str:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={"num_ctx": 2048, "num_predict": max_output_tokens},
        think=False,
    )
    return response["message"]["content"].strip()


def ask_llm(prompt: str, *, max_output_tokens: int = 800, system_instruction: Optional[str] = None) -> str:
    """Provider-agnostic text generation with an optional local fallback."""
    errors = []
    if PROVIDER == "openai":
        try:
            return _openai_text(prompt, max_output_tokens, system_instruction)
        except Exception as exc:
            errors.append(f"OpenAI: {exc}")
            try:
                return _ollama_text(prompt, max_output_tokens, system_instruction)
            except Exception as local_exc:
                errors.append(f"Ollama fallback: {local_exc}")
    else:
        try:
            return _ollama_text(prompt, max_output_tokens, system_instruction)
        except Exception as exc:
            errors.append(f"Ollama: {exc}")
            if _openai_client:
                try:
                    return _openai_text(prompt, max_output_tokens, system_instruction)
                except Exception as api_exc:
                    errors.append(f"OpenAI fallback: {api_exc}")
    raise RuntimeError(" | ".join(errors))


def ask_llm_json(prompt: str, schema: Type[T]) -> T:
    """Return a Pydantic-validated structured response."""
    if PROVIDER == "openai":
        try:
            if not _openai_client:
                raise RuntimeError("OPENAI_API_KEY is not configured.")
            response = _openai_client.responses.parse(
                model=OPENAI_MODEL,
                input=prompt,
                text_format=schema,
            )
            if response.output_parsed is None:
                raise RuntimeError("OpenAI returned no structured result.")
            return response.output_parsed
        except Exception as exc:
            if _openai_client is None:
                raise RuntimeError(str(exc)) from exc
            # Do not silently switch structured planning to unconstrained text.
            try:
                # Ollama JSON mode fallback still gets validated by Pydantic.
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    format=schema.model_json_schema(),
                    options={"num_ctx": 2048, "num_predict": 350},
                    think=False,
                )
                return schema.model_validate_json(response["message"]["content"])
            except Exception as fallback_exc:
                raise RuntimeError(f"Structured planning failed: OpenAI={exc}; Ollama={fallback_exc}") from fallback_exc

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=schema.model_json_schema(),
        options={"num_ctx": 2048, "num_predict": 350},
        think=False,
    )
    return schema.model_validate_json(response["message"]["content"])
