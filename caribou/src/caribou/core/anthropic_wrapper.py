"""
Lightweight Anthropic client wrapper that mimics the subset of the OpenAI
chat API used by CARIBOU. Exposes a `.chat.completions.create(...)` method
that returns an object shaped like the OpenAI response:

    resp.choices[0].message.content
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import anthropic


class AnthropicClient:
    """
    Example:
        client = AnthropicClient(api_key="sk-ant-...", model="claude-sonnet-4-5-20250929")
        resp = client.chat.completions.create(model="claude-sonnet-4-5-20250929", messages=[...])
        print(resp.choices[0].message.content)
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_output_tokens: int = 1024,
        base_url: Optional[str] = None,
    ):
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**client_kwargs)
        self._default_model = model
        self._max_output_tokens = max_output_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat_create))

    def _chat_create(
        self,
        *,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **_: Any,
    ):
        system_parts: List[str] = []
        converted: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(str(content))
                continue
            converted.append({"role": role if role in ("assistant", "user") else "user", "content": content})

        system_prompt = "\n\n".join(system_parts) if system_parts else None

        response = self._client.messages.create(
            model=model or self._default_model,
            system=system_prompt,
            messages=converted,
            temperature=temperature,
            max_tokens=max_output_tokens or self._max_output_tokens,
        )

        text_chunks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        content = "".join(text_chunks)

        message = SimpleNamespace(content=content, role="assistant")
        choice = SimpleNamespace(message=message, index=0, finish_reason=getattr(response, "stop_reason", "stop"))
        return SimpleNamespace(choices=[choice])
