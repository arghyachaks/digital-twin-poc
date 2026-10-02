"""
LLM provider selection (configure in .env, see .env.example).

LLM_PROVIDER = auto | ollama | anthropic | openai | none
  auto  -> local Ollama if it is running (free, private), else Anthropic / OpenAI if a key is set, else none.
With 'none' the assistant still answers in offline mode using the same plant tools (no LLM reasoning).
"""
import json
import os
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# models that support tool calling, best first
OLLAMA_PREFERENCE = ["qwen3", "qwen2.5", "llama3.1", "llama3.2", "llama3.3", "ministral", "mistral-nemo", "mistral", "granite3",
                     "command-r", "firefunction", "hermes3", "gpt-oss", "gemma3"]


def ollama_models():
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=1.5) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return None


def pick_ollama_model(models):
    want = os.getenv("OLLAMA_MODEL")
    if want:
        return want
    for pref in OLLAMA_PREFERENCE:
        for m in models:
            if m.split(":")[0].startswith(pref):
                return m
    return models[0] if models else None


def make_llm():
    """Returns (chat_model or None, description)."""
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    temp = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider in ("auto", "ollama"):
        models = ollama_models()
        if models:
            model = pick_ollama_model(models)
            try:
                from langchain_ollama import ChatOllama
                return ChatOllama(model=model, base_url=OLLAMA_URL, temperature=temp), f"Ollama · {model}"
            except ImportError:
                pass
        elif provider == "ollama":
            return None, ("Ollama not reachable (start the Ollama app)" if models is None
                          else "Ollama has no models (run: ollama pull qwen2.5:7b)")

    if provider in ("auto", "anthropic") and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        return ChatAnthropic(model=model, temperature=temp, max_tokens=1500), f"Anthropic · {model}"

    if provider in ("auto", "openai") and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temp), f"OpenAI · {model}"

    return None, "Offline mode (no LLM configured)"
