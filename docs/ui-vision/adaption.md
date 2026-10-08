# Adapting CARIBOU to the Workbench UI

This note says how the current code base can grow the workbench view described in [README.md](README.md) and mocked in `workbench-demo.html`, while the existing CLI text view and web chat keep working unchanged. It comes from a read-only sweep of the `AddingEvaluatorAgents` branch (as of `f57bab2`, 2026-10-08), which is ahead of the main this folder was branched from. It includes work items, the briefing conversation and evaluator agents. All paths are under `caribou/src/caribou/` unless they start with `frontend/`.

## Goal

One source of truth, two views:

```
runner.py (CLI) ─┐                                    ┌─ CLI text view (default unchanged; opt-in /blocks, /explain)
                 ├─ shared emitter ─► event log + blocks/ ─┤
streaming_runner ┘   (stamps block_id)                └─ web SessionStore ─┬─ chat view (unchanged)
                                                                           └─ workbench view (block-graph reducer)
```

Everything new is additive: new keys on existing event payloads, new event types, and new stores. The chat and the CLI ignore keys they do not know.

## The main finding: there is no single source of truth yet

The README assumes one event stream that both views can read. Today there are three:

| Loop | Events | Persisted |
|---|---|---|
| CLI, `execution/runner.py` | Prints to the Rich console. It emits `RunnerEvent`s (`caribou.runner_event.v1`), but `cli/run_cli.py:220` passes no `event_callback` | Only the chat log or notebook at the end |
| Web, `server/streaming_runner.py` (described at line 8 as "a parallel implementation to execution/runner.py") | Untyped `{type, session_id, turn, timestamp, data}` dicts with different names (`token`, `message_complete`, `status_change`, …) | `session.events` in `session.json`, which is lossy (see bug 2) |
| Control plane, `control/agent_workload.py` | `RunnerEvent` mapped into the typed `Event` journal by `_event_recorder` | Yes, `ExperimentStore` |

The two loops also behave differently:

- `action_id` is `{run}:turn:{t}:block:{i}` in one and `{session}:{t}:{i}` in the other.
- The CLI loop runs only the first code block in each message, by design. The web loop runs all of them.

So the first job (step 0) is to make both loops emit the same events. The two loops already share `execution/work_item_runtime.py`, and the event emitter should be shared the same way.

## Blocks and work items

A **work item** is the *intent*: what should be done, why, by whom, when it counts as done, and whether a reviewer approved it. A **block** is the *implementation* that goes with a work item: how it was actually done.

| | Work item (exists, `execution/work_items.py:141`) | Block (new) |
|---|---|---|
| Answers | What, and why | How, and with what result |
| Holds | title, body, owner, status lifecycle, transitions, reviews, `done_when` | code actions, input/output checkpoints, artifacts, metrics, decisions, checks, story |
| Status | Workflow: In progress, Done, … | Health: pending, running, ok, warn, error, stale |
| Lives in | `work-items/` (Git store) | `blocks/` next to it, using the same Git-store pattern |

How the two relate:

- **One work item has one or more blocks.** A rejected review that sends the item back to In progress produces a new block attempt. A branch is another block for the same work item, and "promote" picks which block counts as the work item's implementation.
- **Attribution.** An event belongs to the block that implements the owner's active work item (`WorkItemStore.blocking_for_owner`, `execution/work_items.py:484`). One shared helper resolves this for both loops.
- **Tickets** are work items anchored to a block, action, artifact or failure. They do not need a third concept.
- **Agents and code samples are attributes of a block, not its identity.** `caribou_single_agent.json` runs the whole chain under one agent, so `agent_switch` cannot mark block boundaries. The demo's Preprocess block has `sample: '(generated)'`.

Proposed `Block` fields:

- **Identity:** `block_id`, `work_item_id`, `session_id`, `run_id?`, `kind` (open vocabulary: load, qc, doublets, …), `title`, `agent_name`, `template_ref?` (code-sample name plus its `code_sample_hashes` entry, or "generated"), `instruction`.
- **Span:** `turn_start`, `turn_end`.
- **Health:** `status`.
- **References only:** `action_ids[]`, `artifact_ids[]`, `metric_record_ids[]`, `failure_ids[]`, `checkpoint_in?`, `checkpoint_out?`.
- **Content:** `decisions[]`, `story?`.
- **Branching:** `parent_block_id?`, `branch_id`.

Proposed `BlockEdge` fields: `from_block`, `to_block`, `contract` (the obs, layers and obsm keys that must be present, plus the producing `artifact_id`), and `status` (ok, or a warning carried down from upstream).

Domain fields that already exist but are mostly unused, and should be reused rather than duplicated (`domain/models.py`):

- `Event.stage`, `correlation_id` (unused) and `causation_event_id` (`:735-748`)
- `Artifact.producer_event_id` and `parent_artifact_ids` (`:778`)
- `FailureRecord.downstream_effect` and `caused_by_failure_id` (`:827`), which model error propagation but are never instantiated
- `Checkpoint.parent_checkpoint_id` (`:923`)
- `MetricRecord.input_artifact_ids` (`:874`)

## Step 0: fix the event foundation

These fixes are worth doing even if the workbench never ships. Every item was checked in the code.

1. **Control-plane runs crash on `work_item_changed`.** `runner.py:1088` and `:1315` emit it. `_event_recorder` (`control/agent_workload.py:591`) has no branch for it and ends in `raise RuntimeError("unsupported runner event")` (`:781`).
2. **The web event log loses its history.**
   - `_on_event` appends every event, tokens included, then trims: 5000 down to 4000 (`server/session_manager.py:1794-1795`, `server/session_state.py:51-59`).
   - `SKIP_PERSIST_TYPES` only skips the save; the trimmed list is still what gets written.
   - In real sessions tokens are about 98% of events, so after about 6 turns the early `agent_switch` and `code_submitted` events are gone.
   - `CodeEventRecord` stores `source=""` (`session_manager.py:1866`), so the code source of an early action can be lost entirely.
3. **The CLI keeps no event log.** `cli/run_cli.py:220` passes no `event_callback`, so a CLI session cannot be shown in the workbench later.
4. **The two loops disagree.** The `action_id` formats differ, events have no sequence number, and the event names differ. The frontend dedupes replays by `(turn, block_index)` because there is no event id.
5. **Errors are swallowed.**
   - `frontend/src/app/core/services/agent-stream.service.ts:162`: `catch { /* malformed message */ }` around both parsing and `_handleEvent`.
   - `frontend/src/app/core/services/session-cache.service.ts:23,38,42`: empty `catch` blocks.
   - `execution/report_generation.py:155-157`: handoff report LLM errors become a console warning and `None`.
6. **The blueprint editor drops `brief_policy`.** `BlueprintContent` and `SaveBlueprintRequest` (`server/models.py:400-410`) and the serializer (`server/routes/config.py:440-445`) do not include it, so saving a blueprint in the UI turns its briefing policy off.
7. **The brief goal is never injected.** `render_work_item_state(..., brief_goal=)` (`execution/work_item_runtime.py:111-123`) supports it, but both callers (`runner.py:968`, `streaming_runner.py:575`) leave it out.
8. **Overwritten artifacts are never re-reported.** `_scan_new_artifacts` (`streaming_runner.py:194-214`) dedupes by filename and scans only the top level, so a re-run that overwrites `qc_report.pdf` never emits a new artifact event.

## Roadmap

Each step keeps the CLI text view and the chat view working.

**Status (2026-10-08):** step 0 and steps 1–3 are implemented on this branch. The block code is `execution/blocks.py`, `GET /api/sessions/{id}/blocks`, the CLI `/blocks` command, `SessionStore`, and `pages/session/workbench/` (open it with `?view=workbench`). Code that runs with no active work item goes to an implicit "no work item" block. Steps 4–6 are not started.

**Step 1: tag every event with its block.**
- Add `block_id` to `code_submitted`, `code_result`, `agent_switch` and `artifact` payloads in both loops, from one shared helper.
- Add `action_id` to artifact events.
- In the control-plane journal, use `Event.stage` and `correlation_id`.
- The JSON schemas set `additionalProperties: false`, so keys go inside `data` or `payload`, not at the top level.

**Step 2: frontend refactor with no behaviour change.**
- Move the event reducers out of `frontend/src/app/pages/session/session.ts` (1,289 lines) into a per-session `SessionStore`. The chat renders from `store.chatItems` exactly as today.
- Pull the inline code card (`session.html` about 338-380) and the tool-call chip (inside `message-bubble`) out into shared components.
- Add reducer tests that replay recorded event logs.

**Step 3: read-only workbench and opt-in CLI views.**
- Web: a toggle on the session page (`session/:id?view=workbench`). It must share the one WebSocket, because `AgentStreamService.connect()` drops any earlier connection.
- A pure `block-graph.reducer` turns events into blocks, edges and status. It uses `block_id` where present and infers blocks from work-item transitions where not.
- The canvas is CSS-grid lanes.
- The side panel reuses `app-code-editor` (readonly), `app-artifact-card` and `MarkdownPipe`.
- CLI: `/blocks` and `/explain <block>` slash commands (registry in `execution/user_commands.py:373-460`), and an offline `caribou run blocks <session_dir>`. The default output is unchanged.

**Step 4: status, adaptors and error propagation.**
- `block_check` events: `{status: pass|warn|fail, checks: [{contract_field, ok, evidence}], propagate}`, produced on the pattern of the evaluator's strict review contract (`execution/evaluation.py:191-221`).
- `depends_on` edges.
- Downstream status is the worse of a block's own status and any propagating upstream status. This is a pure projection; the evaluator does not walk the graph.
- Adaptor contracts live on `BlockEdge`.
- Emit handoff reports, RAG repair and in-turn retries as events (`handoff_report`, `rag`, `recovery`). Today they print to the console only.
- A `decision` event: `{block_id, chosen, alternatives, rationale, evidence_refs}`. The runner emits it for delegations (it knows the action space), and agents emit it through a fenced ```` ```decision ```` block parsed like ```` ```brief ````.

**Step 5: branching.**
- Block-boundary checkpoints keyed by `block_id`, exempt from the keep-the-last-3 rule (`execution/session_recovery.py:25-56`). Each one is a full h5ad copy, so this needs a storage budget.
- `SessionForkRequest` gains `from_checkpoint_id` and `instruction`.
- Fork copies `brief.json` and `blocks/`. Today a fork loses its Goal.
- Replay of a single block's successful actions onto its input checkpoint. Today `literal_replay` can only replay the whole ledger.
- Globals other than `adata` are lost on restore.
- New endpoints: `POST /sessions/{id}/blocks/{bid}/branch`, `GET /sessions/{id}/compare?other=`, and `POST …/promote` and `…/discard`.
- When lanes nest or merge, compute layout with elkjs (layered), adding ngx-vflow if pan or zoom is needed, rather than a hand-rolled layout.

**Step 6: narrative, goal and editing.**
- Goal: `SessionBrief.deliverable` and `done_when`. A v2 schema adds `biological_question` and `sanitized_question`; the model is `frozen` with `extra="forbid"`.
- Story: one summarizer call per completed block, of the same shape as `_generate_agent_report`, returning `{plain, biological, computational, vignette?}`. It is stored as a `block_story` event, and "Explain it to me" reads it.
- The narrative arc is a second call over all `block_story` events plus the brief.
- Edited-code execution needs a new route; nothing calls `exec_code` on user input today.

**New API surface (minimal):**
- `GET /sessions/{id}/blocks` and WS `block_changed`
- `POST /sessions/{id}/work-items` for human-created tickets with an anchor
- The branch, compare, promote and discard routes above
- `GET /sessions/{id}/blocks/{bid}/explain`

## README mapping table, re-checked

| Row | Status now |
|---|---|
| Action block | Still loose. Blocks will be the implementation of a work item (above). |
| Block library (Legos) | Partial. Build it on code samples and blueprints with a metadata sidecar (kind, I/O contract, params) hashed through `BlueprintSpec.code_sample_hashes`. Not a new format. |
| Double-click for literal code | Confirmed, view only. The session code card is a plain `<pre>`. |
| Edit code, LLM customize code | Confirmed none. |
| Open ticket | Out of date. Now mostly covered by work items. Missing: a human-create route and anchors to a block or result. `Backlog` and `Ready` are unreachable because `open()` sets In progress. |
| Readout/plot | Partial. Attribution is by turn only. |
| LLM-driven adaptor | Partly out of date. Delegation transfers work items, and the frozen brief acts as a session-level contract. Handoff reports are still free text and not events. |
| Error propagated from QC | Partly out of date. Review verdicts, stall detection and checkpoint health exist; propagation does not. |
| Goal | Partly covered by `SessionBrief`. Briefing is opt-in. |
| Biological to Sanitized Question | Partly covered. The briefing rewrites the request into an analyzable commitment, but no field keeps the original question. |
| Story, arch, vignettes, Explain | Confirmed none. |
| Demo `rag` and `recovery` events | Do not exist in the web stream. |

## Open questions

- What should events attach to when work items are disabled, or when no work item is active? Options: require an active work item (and say so loudly), or open an implicit block per agent segment. This should be decided, not defaulted silently.
- Should one work item that spans several pipeline stages be split into several blocks? Should the agent be prompted to open one work item per stage?
- The checkpoint storage budget for pinned block-boundary checkpoints.
- Should the CLI loop's one-code-block-per-turn policy also apply on the web, so that action ids line up?
