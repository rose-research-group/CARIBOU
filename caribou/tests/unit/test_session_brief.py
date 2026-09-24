from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from caribou.execution.session_brief import (
    SESSION_BRIEF_SCHEMA,
    BriefParseError,
    BriefPolicy,
    SessionBrief,
    parse_brief_block,
    resolve_brief_policy,
)


def _kwargs(**overrides):
    base = dict(
        deliverable="Ship the parser",
        in_scope=["parser module"],
        out_of_scope=["the UI"],
        done_when=["tests pass"],
        created_at=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return base


def test_valid_brief_round_trips() -> None:
    brief = SessionBrief(**_kwargs())
    assert brief.schema_version == SESSION_BRIEF_SCHEMA
    restored = SessionBrief.model_validate_json(brief.model_dump_json())
    assert restored == brief


def test_empty_deliverable_is_rejected() -> None:
    with pytest.raises(ValidationError, match="deliverable"):
        SessionBrief(**_kwargs(deliverable="   "))


def test_empty_in_scope_is_rejected() -> None:
    with pytest.raises(ValidationError, match="in_scope"):
        SessionBrief(**_kwargs(in_scope=[]))


def test_empty_done_when_is_rejected() -> None:
    with pytest.raises(ValidationError, match="done_when"):
        SessionBrief(**_kwargs(done_when=[]))


def test_overlapping_scope_is_rejected() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        SessionBrief(**_kwargs(in_scope=["parser"], out_of_scope=["parser"]))


def test_brief_is_frozen() -> None:
    brief = SessionBrief(**_kwargs())
    with pytest.raises(ValidationError):
        brief.deliverable = "Ship something else"


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SessionBrief(**_kwargs(), unexpected_field="nope")


def test_resolve_brief_policy_none_override_keeps_blueprint_default() -> None:
    blueprint = BriefPolicy(enabled=False, mode="context")
    assert resolve_brief_policy(blueprint, None) == blueprint

    enabled_blueprint = BriefPolicy(enabled=True, mode="seed_item")
    assert resolve_brief_policy(enabled_blueprint, None) == enabled_blueprint


def test_resolve_brief_policy_off_override_disables_even_an_enabled_blueprint() -> None:
    blueprint = BriefPolicy(enabled=True, mode="seed_item", require_confirmation=False)
    resolved = resolve_brief_policy(blueprint, "off")
    assert resolved.enabled is False
    # Non-enabled fields are preserved from the blueprint, not reset to defaults.
    assert resolved.mode == "seed_item"
    assert resolved.require_confirmation is False


def test_resolve_brief_policy_override_enables_a_blueprint_with_no_brief_policy() -> None:
    # The exact scenario the session-creation override exists for: a
    # blueprint that never declared brief_policy at all still gets one.
    blueprint = BriefPolicy()
    assert blueprint.enabled is False

    resolved = resolve_brief_policy(blueprint, "seed_item")
    assert resolved.enabled is True
    assert resolved.mode == "seed_item"
    assert resolved.require_confirmation == blueprint.require_confirmation

    resolved_context = resolve_brief_policy(blueprint, "context")
    assert resolved_context.enabled is True
    assert resolved_context.mode == "context"


def test_parse_brief_block_valid_json_sets_provenance_not_the_agent() -> None:
    raw = (
        '{"deliverable": "Ship it", "in_scope": ["a"], "out_of_scope": [], '
        '"done_when": ["tests pass"], "created_by": "spoofed", '
        '"created_at": "2020-01-01T00:00:00Z"}'
    )
    brief = parse_brief_block(raw, created_by="master_agent")
    assert brief.deliverable == "Ship it"
    assert brief.created_by == "master_agent"
    assert brief.created_at.year >= 2026


def test_parse_brief_block_rejects_malformed_json() -> None:
    with pytest.raises(BriefParseError, match="not valid JSON"):
        parse_brief_block("{not json", created_by="master_agent")


def test_parse_brief_block_rejects_a_non_object_payload() -> None:
    with pytest.raises(BriefParseError, match="JSON object"):
        parse_brief_block("[1, 2, 3]", created_by="master_agent")


def test_parse_brief_block_surfaces_schema_validation_errors() -> None:
    raw = '{"deliverable": "", "in_scope": [], "out_of_scope": [], "done_when": []}'
    with pytest.raises(BriefParseError, match="failed validation"):
        parse_brief_block(raw, created_by="master_agent")
