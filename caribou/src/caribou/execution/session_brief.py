"""The session brief: a commitment authored before execution and frozen on
acceptance (WS-5 of the implementation brief).

This is an execution-plane module, not `caribou/domain/` — it borrows the
domain's discipline (strict, frozen, versioned) but a `SessionBrief` is not
an `ExperimentSpec` and is not touched by the experiment control plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

SESSION_BRIEF_SCHEMA = "caribou.session_brief.v1"
BriefMode = Literal["context", "seed_item"]


@dataclass(frozen=True)
class BriefPolicy:
    enabled: bool = False
    mode: BriefMode = "context"
    require_confirmation: bool = True

    @classmethod
    def from_dict(cls, raw: object) -> "BriefPolicy":
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("brief_policy must be an object")
        enabled = raw.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("brief_policy.enabled must be a boolean")
        mode = raw.get("mode", "context")
        if mode not in {"context", "seed_item"}:
            raise ValueError("brief_policy.mode must be 'context' or 'seed_item'")
        require_confirmation = raw.get("require_confirmation", True)
        if not isinstance(require_confirmation, bool):
            raise ValueError("brief_policy.require_confirmation must be a boolean")
        return cls(
            enabled=enabled, mode=mode, require_confirmation=require_confirmation
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "require_confirmation": self.require_confirmation,
        }


SessionBriefModeOverride = Optional[Literal["off", "context", "seed_item"]]


def resolve_brief_policy(
    blueprint_policy: "BriefPolicy", override: SessionBriefModeOverride
) -> "BriefPolicy":
    """Apply a session-creation-time override to a blueprint's brief_policy.

    `override=None` means the session didn't ask for one — the blueprint's
    own default stands unchanged. This is what lets a session opt into
    briefing (or explicitly opt out) even when its blueprint has no
    `brief_policy` block, or has a different default, without editing the
    blueprint itself.
    """
    if override is None:
        return blueprint_policy
    if override == "off":
        return BriefPolicy(
            enabled=False,
            mode=blueprint_policy.mode,
            require_confirmation=blueprint_policy.require_confirmation,
        )
    return BriefPolicy(
        enabled=True,
        mode=override,
        require_confirmation=blueprint_policy.require_confirmation,
    )


class SessionBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["caribou.session_brief.v1"] = SESSION_BRIEF_SCHEMA
    deliverable: str
    in_scope: List[str]
    out_of_scope: List[str] = []
    done_when: List[str]
    precedent: List[str] = []
    interface_delta: str = "none"
    risks: List[str] = []
    review_class: Literal["routine", "shared", "scientific", "docs"] = "routine"
    source: Optional[str] = None
    created_at: datetime
    created_by: str = "human"

    @field_validator("deliverable")
    @classmethod
    def _deliverable_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("deliverable must be non-empty")
        return value

    @field_validator("in_scope")
    @classmethod
    def _in_scope_non_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("in_scope must have at least one entry")
        return value

    @field_validator("done_when")
    @classmethod
    def _done_when_non_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("done_when must have at least one entry")
        return value

    @model_validator(mode="after")
    def _scope_disjoint(self) -> "SessionBrief":
        overlap = set(self.in_scope) & set(self.out_of_scope)
        if overlap:
            raise ValueError(
                f"in_scope and out_of_scope overlap: {sorted(overlap)}"
            )
        return self


def render_briefing_prompt() -> str:
    """The briefing-phase instruction appendix — parallels
    `work_items.render_work_item_prompt`: both are standalone appendix
    strings usable either inside `Agent.get_full_prompt` or appended
    directly to an already-built system prompt (as `runner.py` does for
    `render_work_item_prompt`).
    """
    return (
        "\n\n**BRIEFING PHASE.** Before any work starts, interview the human "
        "about what they want, then propose a bounded commitment. Do not "
        "write Python code and do not use work-item commands yet — both are "
        "disabled until the brief is accepted. When you have enough to "
        "propose a brief, emit ONLY this fenced block, alone in its own "
        "message, with no other prose:\n\n"
        "```brief\n"
        "{\n"
        '  "deliverable": "...",\n'
        '  "in_scope": ["..."],\n'
        '  "out_of_scope": ["..."],\n'
        '  "done_when": ["..."]\n'
        "}\n"
        "```\n\n"
        "Optional fields: `precedent`, `interface_delta`, `risks`, "
        '`review_class` ("routine"|"shared"|"scientific"|"docs"), `source`. '
        "The human will accept, reject, or edit your proposal — you cannot "
        "finalize it yourself. If rejected or edited, revise and re-propose."
    )


class BriefParseError(ValueError):
    """The agent's ```brief block failed to parse or validate.

    `feedback` is ready to append to history verbatim for the repair loop
    (WS-5.3): "parse the block, validate against SessionBrief, and on
    failure append a system message with the validation errors for repair".
    """

    def __init__(self, feedback: str) -> None:
        super().__init__(feedback)
        self.feedback = feedback


def parse_brief_block(raw: str, *, created_by: str) -> SessionBrief:
    """Parse and validate a brief block's raw JSON content (the text
    `extract_labeled_block(msg, "brief")` returned) into a `SessionBrief`.
    `created_at`/`created_by`/`schema_version` are the harness's to set, not
    the agent's — they're stripped from the agent's JSON if present and
    always set here, so a stale or spoofed value in the block can't stick.

    Raises `BriefParseError` with agent-facing feedback on any failure.
    """
    import json

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BriefParseError(
            f"Brief block is not valid JSON: {exc}. Resend a corrected "
            "```brief block with the same fields."
        ) from exc
    if not isinstance(payload, dict):
        raise BriefParseError(
            "Brief block must be a JSON object, not a list or scalar."
        )
    payload = dict(payload)
    payload.pop("schema_version", None)
    payload.pop("created_at", None)
    payload.pop("created_by", None)
    payload["created_at"] = datetime.now(timezone.utc)
    payload["created_by"] = created_by
    try:
        return SessionBrief(**payload)
    except Exception as exc:  # pydantic ValidationError, or a bad field type
        raise BriefParseError(
            f"Brief block failed validation: {exc}. Resend a corrected "
            "```brief block."
        ) from exc
