# CARIBOU UI Vision

This folder holds the plan for the interface we want to build to replace the current chatbot, based on a whiteboard sketch (not kept in the repo). `workbench-demo.html` is a clickable mock of it. This note ties each part of the sketch to what CARIBOU does today, and lists what is missing. It comes from a sweep of the repo and manuscript done on 2026-10-07. Items marked "unverified" were not checked directly.

## The idea

Today a user chats with a team of LLM agents, and the analysis is a conversation. The sketch turns the same analysis into a visible structure with three layers:

1. **Action block layer (yellow, green, red).** The analysis as a chain of blocks (QC, preprocess, integration) with readouts and plots hanging off them.
2. **Evidence of LLM decisions (grey and red).** LLM-driven adaptors sit between blocks, and errors are shown flowing downstream from the block that caused them.
3. **Biological narrative (blue).** Story elements, a narrative arch, a communication layer and vignettes, reachable from an "Explain it to me" action.

A Goal (green, right side) anchors everything: a Biological Question is turned into a Sanitized Question.

## Sketch element to CARIBOU today

| Sketch element | Closest thing in CARIBOU | Fit |
|---|---|---|
| Action block (QC, preprocess, integration) | An `Agent` in a blueprint, or a `code_samples/` file such as `QC_Inspection.py`, `Doublets.py`, `Integrate_Harmony.py` | Loose. Agents are not typed, ordered steps. |
| Block library (Community, Personal, Generative Legos) | `code_samples/` in the package and in `CARIBOU_HOME`; presets in `control/presets.py` | Partial. No sharing or generation tier. |
| Double-click for substeps | Nothing structural; `AgentActionSpace.past_actions` and code events after the fact | None |
| Double-click for literal code, code window | Code events with source, stdout and stderr; the `code-editor` component | View only |
| Edit code, LLM customize code | `exec_code` exists in the sandbox but is not exposed to user edits | None |
| Open ticket off a code window or result | `FailureRecord` and `FailureDisposition.investigate` | Loose |
| Readout/plot | `Artifact` records and plot classification in `server/streaming_runner.py`; `MetricRecord` | Partial. Not tied to a producing step. |
| LLM-driven adaptor | Agent handoff reports (`execution/report_generation.py`) and delegation commands | Closest match, but free text, not a typed contract |
| Error propagated from QC | Only `consecutive_failures` for the current run | None |
| Goal | `--prompt` in auto mode; `ExperimentSpec.question` and `hypothesis` | Partial |
| Biological to Sanitized Question | Nothing found | None |
| Story elements, narrative arch, communication layer, vignettes, "Explain it to me" | `MemoryManager` context summaries only, which are not user-facing | None |

## What we can reuse

- **Event stream.** The WebSocket at `/ws/sessions/{id}` replays all events on connect, then streams new ones. Events include `agent_switch` (the nearest thing to graph edges), `code_submitted`, `code_result` and `artifact`. A block view can be rebuilt from this.
- **Control plane.** `control/` and `routes/experiments.py` already have runs, events, checkpoints, resume and artifact verification. This is the best fit for a run, step and artifact graph.
- **Sessions.** Persistence, fork, resume and retry already exist.
- **Execution.** The sandbox (`core/sandbox_management.py`, persistent kernel) and the existing code editor and markdown rendering.
- **Blueprints.** JSON agent graphs, plus the blueprint editor page, which edits that graph as a form.

## What is missing

1. **A workflow model.** Today a pipeline is a conversation, and its order lives only in blueprint prompts. We need a block schema (id, type, inputs, outputs, code, status, parent, children), so events and artifacts can be attributed to a block.
2. **A graph canvas.** The frontend is Angular 21 with no graph library.
3. **Typed adaptors.** An adaptor would be a new step kind with a schema between blocks.
4. **Block-level run control.** There is no way to run one block, run an edited block, or re-run downstream.
5. **Dependency edges and per-block status.** This is what error propagation needs.
6. **Tickets,** from code windows and from results.
7. **Goal handling.** A stored goal and a Biological to Sanitized Question step, probably one LLM call.
8. **The narrative layer.** An LLM summarizer over blocks and results, with models and endpoints for story elements, the arch, vignettes and the explain action.

## Where the manuscript backs this up

The Discussion section of the manuscript (`main.tex`, kept outside this repo) makes the case for the error-propagation part of the sketch. It says early-pipeline errors propagate silently, that there is no systematic checkpoint and rollback, and that sessions can be "internally consistent but analytically incorrect". It asks for explicit checkpointing, output validation and automated refinement loops instead of conversational correction. The manuscript also describes the "session artifact system" (reports, ledgers, code snippets, notebook logs) as a foundation for auditability, which is the raw material for the evidence layer.

The manuscript does not mention the web UI, a narrative layer or an explanation layer, so these are new contributions.

## Workflows to prototype on

- **Allen Brain Atlas hippocampus** (85,955 cells, 3 batches): the clearest QC, doublet, normalize, integrate, cluster chain, scored with SCIB. Best for the action block layer.
- **Case 1, QC report:** a single QC block, with the pre and post QC distributions as the readout node.
- **Case 2, batch correction:** the LLM picks Harmony or scVI between QC and integration, which is the adaptor decision to expose.
- **Failure recovery:** the missing QC obs field case (detect, diagnose, generate, execute, continue) shows evidence of LLM decisions. Figure numbers for these are unverified.
- **Case 3, PBMC CellTypist annotation, and Tabula Sapiens large intestine:** biology-facing results that suit the narrative layer.
- **Metadata reconstruction (annotations stripped):** a natural test for Goal and Sanitized Question, since the biology is unknown.

## Demo

`workbench-demo.html` is a clickable mock of the vision. It is one self-contained file with no build step and no server: open it in a browser.

It walks through the Allen Brain Atlas hippocampus run. Agents, code samples, event types and SCIB metrics follow the current repo. Results, plots and the batch 3 QC problem are mocked.

What it shows:

- **Landing view:** the question, a one-sentence answer with one alert, and a single row of steps.
- **Progressive disclosure:** click a step for a summary in the side panel; double-click for substeps; double-click a substep for its code. Decisions, code editing, run controls and activity are collapsed rows.
- **Row labels:** Plots, Story and Paths fold in and out from their labels on the left of the canvas.
- **Adaptors:** the circles between steps, with contract checks and the handoff note.
- **Error propagation:** a QC warning carried forward to every later step.
- **Branching:** select a step, click the Branch tab on it, and say what should differ. Paths can be compared, promoted to main, or discarded.

Deep links for demos and screenshots, added to the URL:

| Hash | State |
|---|---|
| `#block:qc:main:open` | QC selected with substeps and decisions open |
| `#compose:int` | Branch box open on Integration |
| `#branches` | Two extra paths created, compare panel open |
| `#all` | Plots, Story and the full narrative unfolded |

The demo is vanilla HTML, CSS and JS so it can be changed quickly. Colours are the tokens from `frontend/src/styles.scss`. Nothing in it talks to the CARIBOU server yet.

## Open questions

- Should blocks map one-to-one to blueprint agents, to code samples, or to a new step type?
- Is the "Legos" library built on `code_samples/` and blueprints, or a new format?
- Does "Sanitized Question" mean removing identifying or dataset-specific details, or rewriting into an analyzable form? The sketch does not say.
- What does "ticket" connect to (GitHub issues, an internal queue)?
