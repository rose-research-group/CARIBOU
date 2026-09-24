"""Shared DeepSeek model profiles and OpenAI-compatible request policy."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping


DEEPSEEK_API_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_FAST_BACKEND = "deepseek"
DEEPSEEK_FLASH_V41_BACKEND = "deepseek-v4.1"
DEEPSEEK_THINKING_BACKEND = "deepseek-thinking"
DEEPSEEK_FAST_MODEL = "deepseek-v4-flash"
DEEPSEEK_FLASH_V41_MODEL = "deepseek-flash"
DEEPSEEK_THINKING_MODEL = "deepseek-v4-pro"

_THINKING_INCOMPATIBLE_OPTIONS = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
)


@dataclass(frozen=True)
class DeepSeekProfile:
    """An exact, user-selectable DeepSeek execution profile."""

    backend_id: str
    model: str
    display_name: str
    thinking: bool
    reasoning_effort: str | None = None

    def model_parameters(self) -> dict[str, object]:
        """Return provider controls suitable for frozen ExperimentSpec provenance."""

        parameters: dict[str, object] = {"thinking": self.thinking}
        if self.reasoning_effort is not None:
            parameters["reasoning_effort"] = self.reasoning_effort
        return parameters


DEEPSEEK_FAST_PROFILE = DeepSeekProfile(
    backend_id=DEEPSEEK_FAST_BACKEND,
    model=DEEPSEEK_FAST_MODEL,
    display_name="DeepSeek V4 Flash (Quick)",
    thinking=False,
)
DEEPSEEK_FLASH_V41_PROFILE = DeepSeekProfile(
    backend_id=DEEPSEEK_FLASH_V41_BACKEND,
    model=DEEPSEEK_FLASH_V41_MODEL,
    display_name="DeepSeek V4.1 Flash (Quick)",
    thinking=False,
)
DEEPSEEK_THINKING_PROFILE = DeepSeekProfile(
    backend_id=DEEPSEEK_THINKING_BACKEND,
    model=DEEPSEEK_THINKING_MODEL,
    display_name="DeepSeek V4 Pro (Thinking)",
    thinking=True,
    reasoning_effort="high",
)
DEEPSEEK_PROFILES = (
    DEEPSEEK_FAST_PROFILE,
    DEEPSEEK_FLASH_V41_PROFILE,
    DEEPSEEK_THINKING_PROFILE,
)
DEEPSEEK_BACKEND_IDS = tuple(profile.backend_id for profile in DEEPSEEK_PROFILES)
DEEPSEEK_MODEL_IDS = tuple(profile.model for profile in DEEPSEEK_PROFILES)

_PROFILES_BY_BACKEND = {
    profile.backend_id: profile for profile in DEEPSEEK_PROFILES
}
_PROFILES_BY_MODEL = {profile.model: profile for profile in DEEPSEEK_PROFILES}


def is_deepseek_backend(backend: str) -> bool:
    return backend in _PROFILES_BY_BACKEND


def deepseek_profile_for_backend(backend: str) -> DeepSeekProfile:
    try:
        return _PROFILES_BY_BACKEND[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown DeepSeek backend: {backend!r}") from exc


def deepseek_profile_for_model(model: str) -> DeepSeekProfile:
    try:
        return _PROFILES_BY_MODEL[model]
    except KeyError as exc:
        raise ValueError(f"Unsupported DeepSeek model: {model!r}") from exc


class _DeepSeekCompletions:
    """Inject the selected DeepSeek mode into every completion request."""

    def __init__(
        self,
        completions: Any,
        *,
        thinking: bool | None,
        reasoning_effort: str | None,
    ) -> None:
        self._completions = completions
        self._thinking = thinking
        self._reasoning_effort = reasoning_effort

    def create(self, **kwargs: Any) -> Any:
        request = dict(kwargs)
        if self._thinking is not None:
            raw_extra_body = request.get("extra_body")
            if raw_extra_body is None:
                extra_body: dict[str, Any] = {}
            elif isinstance(raw_extra_body, Mapping):
                extra_body = dict(raw_extra_body)
            else:
                raise TypeError("DeepSeek extra_body must be a mapping")
            extra_body["thinking"] = {
                "type": "enabled" if self._thinking else "disabled"
            }
            request["extra_body"] = extra_body

            if self._thinking:
                # DeepSeek documents these sampling controls as unsupported in
                # thinking mode. Removing them also prevents recorded request
                # settings from implying that they affected the response.
                for option in _THINKING_INCOMPATIBLE_OPTIONS:
                    request.pop(option, None)
                if self._reasoning_effort is not None:
                    request["reasoning_effort"] = self._reasoning_effort
            else:
                request.pop("reasoning_effort", None)

        return self._completions.create(**request)


class DeepSeekClient:
    """Small adapter exposing the OpenAI client surface CARIBOU consumes."""

    def __init__(
        self,
        raw_client: Any,
        *,
        thinking: bool | None,
        reasoning_effort: str | None = None,
    ) -> None:
        if reasoning_effort is not None and thinking is not True:
            raise ValueError("DeepSeek reasoning_effort requires thinking mode")
        self._raw_client = raw_client
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.chat = SimpleNamespace(
            completions=_DeepSeekCompletions(
                raw_client.chat.completions,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
            )
        )


def create_deepseek_client(
    api_key: str,
    *,
    profile: DeepSeekProfile | None = None,
    thinking: bool | None = None,
    reasoning_effort: str | None = None,
    max_retries: int | None = None,
) -> DeepSeekClient:
    """Create a DeepSeek client with an explicit completion request policy."""

    if profile is not None:
        if thinking is not None or reasoning_effort is not None:
            raise ValueError("use either a DeepSeek profile or explicit mode settings")
        thinking = profile.thinking
        reasoning_effort = profile.reasoning_effort

    from openai import OpenAI

    client_options: dict[str, object] = {
        "api_key": api_key,
        "base_url": DEEPSEEK_API_BASE_URL,
    }
    if max_retries is not None:
        client_options["max_retries"] = max_retries
    raw_client = OpenAI(**client_options)
    return DeepSeekClient(
        raw_client,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
    )
