"""The endpoint layer — the ONLY module allowed to import the OpenAI SDK.

It exposes three capabilities to the rest of the package:

    chat(messages, ...)        -> streamed or buffered chat completion
    embed(texts)               -> embedding vectors
    transcribe_image(b64, ...) -> vision call (used by the PDF parser)

Everything speaks the OpenAI API: the standard endpoint by default, or any
OpenAI-compatible endpoint via OPENAI_BASE_URL. No other file should construct
a client or hardcode a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

from openai import BadRequestError, OpenAI

from contextual_rag.config import ConfigError, Settings, get_settings

# --- Message / result types ------------------------------------------------

# A chat message is the usual {"role": ..., "content": ...} dict.
Message = dict[str, str]


@dataclass(frozen=True)
class Usage:
    """Token accounting for one chat call.

    `cached_tokens` is the portion of `prompt_tokens` served from the provider's
    automatic prompt-prefix cache (0 if the endpoint doesn't report it). It's what
    makes Contextual Retrieval cheap — the repeated document prefix is mostly cached.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0


@dataclass(frozen=True)
class ChatResult:
    """Return value of a non-streaming chat() call."""

    text: str
    usage: Usage | None
    model: str


@dataclass(frozen=True)
class StreamEvent:
    """One event from a streaming chat() call.

    type="delta"  -> `text` holds the next token(s).
    type="usage"  -> `usage` holds final token counts (arrives once, at the end).
    """

    type: Literal["delta", "usage"]
    text: str = ""
    usage: Usage | None = None


# --- Client construction ----------------------------------------------------

_client: OpenAI | None = None
_client_settings: Settings | None = None


def _build_client(s: Settings) -> OpenAI:
    if s.base_url:
        # OpenAI-compatible endpoint at a custom URL. Keyless gateways still need
        # a non-empty api_key string (the SDK refuses an empty one), so we pass a
        # placeholder the server ignores.
        return OpenAI(base_url=s.base_url, api_key=s.api_key or "not-needed")
    if not s.api_key:
        raise ConfigError(
            "No endpoint configured. Set OPENAI_API_KEY (standard OpenAI), or "
            "OPENAI_BASE_URL for an OpenAI-compatible gateway — via env/.env or "
            "contextual_rag.configure(...)."
        )
    return OpenAI(api_key=s.api_key)


def _get_client() -> OpenAI:
    """One shared client per active Settings; rebuilt after configure()."""
    global _client, _client_settings
    s = get_settings()
    if _client is None or _client_settings is not s:
        _client = _build_client(s)
        _client_settings = s
    return _client


def _to_usage(raw) -> Usage | None:
    if not raw:
        return None
    details = getattr(raw, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0 if details is not None else 0
    return Usage(
        prompt_tokens=raw.prompt_tokens,
        completion_tokens=raw.completion_tokens,
        total_tokens=raw.total_tokens,
        cached_tokens=cached,
    )


# --- Public API ------------------------------------------------------------

def chat(
    messages: Sequence[Message],
    *,
    stream: bool = False,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatResult | Iterator[StreamEvent]:
    """Run a chat completion.

    stream=False -> returns a ChatResult (buffered text + usage).
    stream=True  -> returns an iterator of StreamEvent (deltas, then one usage).

    `model` overrides the configured chat model; the other kwargs override
    generation defaults. Streaming requests opt into usage reporting so callers
    can still show token counts after a live answer.
    """
    s = get_settings()
    model = model or s.chat_model
    temperature = s.chat_temperature if temperature is None else temperature

    if stream:
        return _chat_stream(messages, model, temperature, max_tokens)
    return _chat_buffered(messages, model, temperature, max_tokens)


def _common_kwargs(model, messages, temperature, max_tokens) -> dict:
    """Shared request kwargs. Only include `max_tokens` when actually set —
    newer models (gpt-5, o-series) reject the legacy `max_tokens` param, even as
    null, so sending it unconditionally breaks them.
    """
    kwargs = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def _create(**kwargs):
    """Create a completion, retrying without params the model rejects.

    Reasoning models (gpt-5, o1/o3/o4...) only allow the default temperature and
    take `max_completion_tokens` instead of the legacy `max_tokens`, so on the
    matching 400 we adapt and retry. Generic by design — no per-model table to
    maintain across endpoints.
    """
    client = _get_client()
    for _ in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            msg = str(exc).lower()
            if "max_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                continue
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            raise
    return client.chat.completions.create(**kwargs)


def _chat_buffered(messages, model, temperature, max_tokens) -> ChatResult:
    resp = _create(
        **_common_kwargs(model, messages, temperature, max_tokens),
        stream=False,
    )
    return ChatResult(
        text=resp.choices[0].message.content or "",
        usage=_to_usage(resp.usage),
        model=resp.model,
    )


def _chat_stream(messages, model, temperature, max_tokens) -> Iterator[StreamEvent]:
    stream = _create(
        **_common_kwargs(model, messages, temperature, max_tokens),
        stream=True,
        # Without this, the streamed response omits the usage block entirely.
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        # Usage-only chunks carry no choices; delta chunks carry no usage.
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield StreamEvent(type="delta", text=delta.content)
        if chunk.usage:
            yield StreamEvent(type="usage", usage=_to_usage(chunk.usage))


def transcribe_image(
    image_b64: str,
    prompt: str,
    *,
    model: str | None = None,
    detail: str = "auto",
    max_tokens: int | None = None,
) -> ChatResult:
    """Vision call: send one page image (base64 PNG) + a prompt, get text back.

    Uses the same OpenAI-compatible client as chat(), with the standard image
    content part. temperature=0 for faithful transcription (dropped automatically
    if the model rejects it). `detail` is the vision fidelity ("low"|"high"|"auto"):
    "high" forces max tiling for small/dense text. This is how PDF pages are
    parsed (render → image → here).
    """
    s = get_settings()
    model = model or s.parse_model or s.chat_model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": detail,
                    },
                },
            ],
        }
    ]
    resp = _create(**_common_kwargs(model, messages, 0, max_tokens), stream=False)
    return ChatResult(
        text=resp.choices[0].message.content or "",
        usage=_to_usage(resp.usage),
        model=resp.model,
    )


def list_models() -> list[str]:
    """List model ids available at the configured endpoint (any OpenAI-compatible
    endpoint exposes /v1/models)."""
    resp = _get_client().models.list()
    return sorted(m.id for m in resp.data)


def embed(texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
    """Embed one or more texts, returning a vector per input (order preserved).

    The embedding dimension is whatever the model returns — callers must read it
    dynamically, never assume it, since it differs between models.
    """
    if not texts:
        return []
    resp = _get_client().embeddings.create(
        model=model or get_settings().embed_model,
        input=list(texts),
    )
    # API guarantees results are returned in input order, but sort defensively.
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]
