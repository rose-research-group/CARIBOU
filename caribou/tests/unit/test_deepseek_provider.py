from __future__ import annotations

from types import SimpleNamespace

import pytest

from caribou.core.deepseek import (
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_FAST_MODEL,
    DEEPSEEK_FAST_PROFILE,
    DEEPSEEK_FLASH_V41_MODEL,
    DEEPSEEK_FLASH_V41_PROFILE,
    DEEPSEEK_THINKING_MODEL,
    DEEPSEEK_THINKING_PROFILE,
    DeepSeekClient,
    create_deepseek_client,
    deepseek_profile_for_backend,
    deepseek_profile_for_model,
)
from caribou.server.routes.config import _BACKENDS, _KEY_MAP
from caribou.server.session_setup import build_llm_client, resolve_model_info


class RecordingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return object()


def _raw_client(completions: RecordingCompletions) -> object:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


@pytest.mark.parametrize(
    ("profile", "expected_model"),
    [
        (DEEPSEEK_FAST_PROFILE, DEEPSEEK_FAST_MODEL),
        (DEEPSEEK_FLASH_V41_PROFILE, DEEPSEEK_FLASH_V41_MODEL),
    ],
)
def test_quick_profiles_lock_flash_and_disable_thinking(
    profile: object, expected_model: str
) -> None:
    completions = RecordingCompletions()
    client = DeepSeekClient(
        _raw_client(completions),
        thinking=profile.thinking,
    )

    client.chat.completions.create(
        model=expected_model,
        messages=[{"role": "user", "content": "quick"}],
        temperature=0.0,
        reasoning_effort="max",
    )

    request = completions.calls[0]
    assert request["model"] == expected_model
    assert request["temperature"] == 0.0
    assert "reasoning_effort" not in request
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_thinking_profile_locks_v4_pro_and_high_effort() -> None:
    completions = RecordingCompletions()
    client = DeepSeekClient(
        _raw_client(completions),
        thinking=DEEPSEEK_THINKING_PROFILE.thinking,
        reasoning_effort=DEEPSEEK_THINKING_PROFILE.reasoning_effort,
    )

    client.chat.completions.create(
        model=DEEPSEEK_THINKING_MODEL,
        messages=[{"role": "user", "content": "think"}],
        temperature=0.7,
        top_p=0.9,
        extra_body={"trace_id": "test-trace"},
    )

    request = completions.calls[0]
    assert request["model"] == "deepseek-v4-pro"
    assert "temperature" not in request
    assert "top_p" not in request
    assert request["reasoning_effort"] == "high"
    assert request["extra_body"] == {
        "trace_id": "test-trace",
        "thinking": {"type": "enabled"},
    }


def test_profiles_resolve_from_stable_backend_and_exact_model_ids() -> None:
    assert deepseek_profile_for_backend("deepseek") is DEEPSEEK_FAST_PROFILE
    assert (
        deepseek_profile_for_backend("deepseek-v4.1")
        is DEEPSEEK_FLASH_V41_PROFILE
    )
    assert (
        deepseek_profile_for_backend("deepseek-thinking")
        is DEEPSEEK_THINKING_PROFILE
    )
    assert deepseek_profile_for_model(DEEPSEEK_FAST_MODEL) is DEEPSEEK_FAST_PROFILE
    assert (
        deepseek_profile_for_model(DEEPSEEK_FLASH_V41_MODEL)
        is DEEPSEEK_FLASH_V41_PROFILE
    )
    assert (
        deepseek_profile_for_model(DEEPSEEK_THINKING_MODEL)
        is DEEPSEEK_THINKING_PROFILE
    )

    with pytest.raises(ValueError, match="Unsupported DeepSeek model"):
        deepseek_profile_for_model("deepseek-chat")


@pytest.mark.parametrize(
    ("backend", "expected_model", "expected_thinking"),
    [
        ("deepseek", DEEPSEEK_FAST_MODEL, False),
        ("deepseek-v4.1", DEEPSEEK_FLASH_V41_MODEL, False),
        ("deepseek-thinking", DEEPSEEK_THINKING_MODEL, True),
    ],
)
def test_web_session_builder_uses_shared_profiles(
    backend: str,
    expected_model: str,
    expected_thinking: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_calls: list[dict[str, object]] = []
    completions = RecordingCompletions()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: openai_calls.append(kwargs) or _raw_client(completions),
    )

    client, model = build_llm_client(
        SimpleNamespace(llm_backend=backend, ollama_model=None)
    )

    assert isinstance(client, DeepSeekClient)
    assert model == expected_model
    assert client.thinking is expected_thinking
    assert openai_calls == [
        {
            "api_key": "test-deepseek-key",
            "base_url": DEEPSEEK_API_BASE_URL,
        }
    ]


def test_backend_discovery_exposes_exact_deepseek_profiles() -> None:
    deepseek_backends = [backend for backend in _BACKENDS if backend.provider == "deepseek"]

    assert [backend.id for backend in deepseek_backends] == [
        "deepseek",
        "deepseek-v4.1",
        "deepseek-thinking",
    ]
    assert [backend.model_name for backend in deepseek_backends] == [
        DEEPSEEK_FAST_MODEL,
        DEEPSEEK_FLASH_V41_MODEL,
        DEEPSEEK_THINKING_MODEL,
    ]
    assert [backend.thinking for backend in deepseek_backends] == [False, False, True]
    assert _KEY_MAP["deepseek"] == "DEEPSEEK_API_KEY"
    assert _KEY_MAP["deepseek-v4.1"] == "DEEPSEEK_API_KEY"
    assert _KEY_MAP["deepseek-thinking"] == "DEEPSEEK_API_KEY"


@pytest.mark.parametrize(
    ("backend", "expected_model", "expected_parameters"),
    [
        ("deepseek", DEEPSEEK_FAST_MODEL, {"thinking": False}),
        (
            "deepseek-v4.1",
            DEEPSEEK_FLASH_V41_MODEL,
            {"thinking": False},
        ),
        (
            "deepseek-thinking",
            DEEPSEEK_THINKING_MODEL,
            {"thinking": True, "reasoning_effort": "high"},
        ),
    ],
)
def test_user_facing_model_record_is_exact_and_versioned(
    backend: str,
    expected_model: str,
    expected_parameters: dict[str, object],
) -> None:
    record = resolve_model_info(
        SimpleNamespace(llm_backend=backend, ollama_model=None)
    )

    assert record is not None
    assert record.provider == "deepseek"
    assert record.model == expected_model
    assert record.parameters == expected_parameters


def test_client_creation_rejects_conflicting_profile_settings() -> None:
    with pytest.raises(ValueError, match="either a DeepSeek profile"):
        create_deepseek_client(
            "unused",
            profile=DEEPSEEK_FAST_PROFILE,
            thinking=True,
        )
