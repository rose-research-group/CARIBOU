# External-agent CLI contract

CARIBOU exposes a non-interactive lifecycle for Codex, Claude, scripts, and
other long-running callers. Non-streaming commands emit exactly one JSON object
on stdout. Event reads emit resumable JSON Lines. Semantic failures use the
stable exit codes reported by `capabilities`.

The shortest validated journey uses the deterministic control-plane probe:

```bash
caribou capabilities --json
caribou schema experiment --json
caribou experiment init --output experiment.yaml --json
caribou experiment validate experiment.yaml --json
caribou experiment plan experiment.yaml --json
caribou experiment submit experiment.yaml \
  --idempotency-key lifecycle-smoke-001 --json
```

`experiment init` writes a valid lifecycle-smoke spec with a fresh spec ID and
the executing CARIBOU repository, branch, commit, and worktree state. It refuses
to guess unresolved provenance or replace an existing file unless
`--overwrite` is explicit. Use it to prove the automation journey, then replace
the smoke inputs, condition, evaluator, resources, and scientific question with
frozen real values for an analysis.

Submission returns after durably queuing the run and launching a detached
worker. Preserve the returned `run_ids[0]`, then reconnect from any later CLI
process:

```bash
caribou run status RUN_ID --json
caribou run events RUN_ID --after 0 --format jsonl
caribou artifact list RUN_ID --json
caribou artifact verify RUN_ID --json
caribou artifact fetch RUN_ID ARTIFACT_ID --output ./result.json --json
caribou experiment compare EXPERIMENT_ID --json
```

Use the last event cursor as the next `--after` value to avoid duplicates.
`experiment compare` is a deterministic, read-only summary of condition
interventions, logical leaf attempts, outcomes, resume lineage, event cursors,
and record inventories. Superseded checkpoint source attempts remain visible but
are not counted twice. The current comparison does not execute metric evaluators
or claim a scientific aggregate; `metric_values_aggregated` remains false until
that separate evaluation path is implemented.
Cancellation is durable and idempotent:

```bash
caribou run cancel RUN_ID --reason "caller stopped the experiment" --json
```

Agent runs can also stop cooperatively at the next completed-turn boundary:

```bash
caribou run checkpoint RUN_ID \
  --idempotency-key checkpoint-RUN_ID-turn-2 \
  --reason "preserve this completed analysis turn" --json
caribou run checkpoints RUN_ID --json
```

The worker finishes the current model response and its declared actions before
publishing the checkpoint. It then preserves the source attempt as terminal
`resumable`; it does not begin another model turn. A complete checkpoint links
hash-verified dataset, full message history, runner/agent state, executed-action
ledger, and artifact-frontier components. Resume creates a new immutable attempt
rather than reopening the source:

```bash
caribou run resume RUN_ID --from-checkpoint latest \
  --idempotency-key resume-RUN_ID-turn-2 --json
```

Retrying the exact resume request returns the same child and does not duplicate
workload execution. Local workers must claim the one durable process handle before
changing run state, invoking a model, or executing generated code. Selecting the
same checkpoint with another key fails because the initial slice does not support
branching. The child retains the frozen experiment,
model, code, container, budget, and total turn limit; it binds the checkpointed
AnnData artifact at `/workspace/dataset.h5ad`, restores the full-history cursor and
current agent, and starts with the next logical turn. Historical resumable attempts
remain in the ledger but do not prevent a successful leaf attempt from completing
the experiment.

This recovery contract currently supports only full-history memory and one
cooperative checkpoint per attempt. It is not a Python-process snapshot: only the
declared AnnData global `adata`, messages, runner counters, current agent, action
ledger, and artifact frontier are restored. A hard loss during a provider call,
between a provider receipt and durable response content, or midway through code
execution is ambiguous and is not automatically replayed. Scheduler preemption,
SIGKILL, episodic/report memory, arbitrary Python globals, and branching require
separate validation before they can be claimed.

Retry submission with the same idempotency key and unchanged specification to
recover the original experiment and run IDs without launching another worker.
Reusing the key with different content fails with a typed conflict. A caller
that records the plan hash may also pass `--expected-plan-hash` at submission
to reject configuration drift.

The lifecycle smoke adapter validates only local submission, persistence,
events, cancellation, and artifact handling. It does not invoke a model,
container, biological analysis, or Slurm.

The web experiment wizard is a human-facing adapter to this same contract. Its
authenticated `GET /api/control/presets` endpoint discovers bounded pilot
patterns, and `POST /api/control/presets/{preset_id}/resolve` freezes the selected
dataset, code commit, blueprint, prompts, code samples, RAG corpus, container,
model, resources, and evaluator declaration into an `ExperimentSpec`. Resolution
does not submit work. The browser next calls the canonical plan endpoint and then
the canonical idempotent submit endpoint with the returned plan hash. A dirty
checkout is rejected, and the first resolution of a large container can take time
while its content hash is computed. Presets are convenience inputs, not validated
biological protocols or evidence of benchmark performance.

Experiment specification v2 additionally freezes one evaluator agent reference
and one exact evaluator model for the whole experiment. The evaluator is distinct
from each condition's worker model and from metric evaluator artifacts. Submitted
specifications are immutable: use `caribou experiment clone` (or **Clone model**
in the web run monitor), optionally recording a model-change reason, then validate,
plan, and submit the new draft. Version-1 specifications remain readable but do
not acquire an inferred evaluator.

Interactive CLI and web sessions default the evaluator model to the worker model
for compatibility. A separate evaluator can be selected at creation. Web sessions
may change only the evaluator model in place using a revision-checked request;
the change and optional reason are recorded as an event and system message. In the
interactive CLI, `/evaluator status` inspects the binding and `/evaluator model
--llm PROVIDER --model MODEL_ID [--reason TEXT]` changes it. Worker model changes
continue to require a fork, where the optional reason is also retained.

The separately tested `agent_path_smoke` adapter crosses the real control
service, detached worker, existing agent runner, blueprint delegation, command
parser, event journal, and artifact store. Its model client and sandbox are
deterministic test boundaries: it does not validate an external model,
Apptainer/Singularity, biological correctness, or scheduler execution.

The smoke adapter accepts a dirty development worktree only when the frozen code
identity records `dirty: true`; its exact HEAD commit must still match. The real
`caribou_agent` adapter continues to reject every dirty worktree.

The initial `caribou_agent` adapter is implemented for a deliberately narrow
real pilot. It requires a clean exact Git commit, one hash-pinned local input,
a hash-pinned Apptainer/Singularity image, network-disabled generated code,
full-history memory, a frozen corpus whenever the blueprint enables RAG, no
external blueprint tools, no ambient cache or custom mounts, and an external
OpenAI or DeepSeek model identified
by exact model ID. Finite budget limits, model tuning fields, declared
container-runtime versions, retry policies, and implicit CellTypist caches are
rejected rather than ignored. Provider requests receive the remaining frozen
session deadline. The experiment metric definitions remain preregistered
metadata; this adapter does not execute their evaluator artifacts or enforce
the declared local CPU, memory, and storage maxima.

For DeepSeek V4, frozen model parameters may additionally contain `thinking`
(boolean) and `reasoning_effort` (`high` or `max`; valid only when thinking is
enabled). The preset resolver locks `deepseek-v4-flash` and `deepseek-flash`
(the `deepseek-v4.1` backend's model id — DeepSeek now serves V4.1 Flash
under this id, not a separate "v4.1" string) to quick mode and
`deepseek-v4-pro` to thinking mode at high effort so the effective request mode
is preserved alongside the exact model ID. Standard CLI session reports created
with `--make-report` also record these values as `model` and
`model_parameters`.

When the worker remains alive through receipt persistence, every completed or
failed OpenAI-compatible SDK attempt made by this adapter produces a
hash-verifiable `provider_call_receipt` artifact before response content is
used. Query its strict consumer contract with:

```bash
caribou schema provider-call-receipt --json
```

Receipts contain only whitelisted provider and request identifiers, the
requested and provider-returned model names, timing, finish status, and token
counts when supplied by the provider. They never contain prompts, response
content, raw error text or bodies, headers, URLs, clients, or credentials.
Missing usage remains `null`, never zero. Provider cost is explicitly
unavailable, and the run's budget counters are not yet reconciled from these
receipts. Do not infer resource enforcement, billed cost, evaluator execution,
or biological validity from a successful run. A hard process or node loss
during a provider request can leave a billed request without a receipt; this
slice does not yet implement pre-call intent logging or crash reconciliation.

Query `capabilities` rather than assuming later execution surfaces are
validated.
