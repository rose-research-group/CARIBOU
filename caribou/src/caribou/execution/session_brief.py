"""The session brief: a commitment authored before execution and frozen on
acceptance (WS-5 of the implementation brief).

This is an execution-plane module, not `caribou/domain/` — it borrows the
domain's discipline (strict, frozen, versioned) but a `SessionBrief` is not
an `ExperimentSpec` and is not touched by the experiment control plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
