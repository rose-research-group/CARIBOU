# CARIBOU: The Open-source Language Agent Framework 🚀

**CARIBOU is a multi-agent LLM framework for bioinformatics on GPU and HPC environments.** It orchestrates teams of specialized AI agents that collaborate inside secure sandboxes (Docker or Singularity) to perform complex analyses on single-cell and spatial transcriptomics data.

![CARIBOU web UI — session view](docs/images/web-ui-session-chat.png)

CARIBOU ships with two interfaces:

- **CLI** — interactive and automated runs directly from the terminal, designed for scripting and benchmarking.
- **Web UI** — a full Angular frontend served by a FastAPI backend, accessible from any browser via SSH tunnel or Open OnDemand. Sessions persist across server restarts.

---

## Key Features

- **Multi-Agent Blueprints** — define agents, their specialized prompts, and delegation chains in a simple JSON config.
- **Secure Sandboxing** — all agent-generated code runs in isolated Docker or Singularity containers.
- **Interactive & Auto Modes** — turn-by-turn interactive chat for debugging, or fully automated multi-turn runs for benchmarking.
- **Web Frontend** — real-time streaming UI with markdown rendering, syntax-highlighted code blocks, inline plot previews with lightbox, and persistent session history.
- **Multi-LLM Support** — unified interface for OpenAI (GPT-4o), Anthropic (Claude), DeepSeek, and local Ollama models.
- **Data Curation** — browse and download single-cell datasets from the CZI CELLxGENE Census.
- **Retrieval-Augmented Generation** — agents can query a knowledge base of bioinformatics function signatures when they hit API errors.

---

## Installation

### Prerequisites

1. **Python 3.9+**
2. **A sandbox backend:** Docker (daemon must be running) or Singularity/Apptainer
3. **Node.js 18+** *(only required to build the web frontend)*

### Install from Source

```bash
git clone https://github.com/OpenTechBio/CARIBOU
cd CARIBOU
pip install -e caribou/
```

---

## Quick Start (CLI)

### Step 1: Configure your API key

```bash
caribou config set-openai-key     "sk-..."
caribou config set-anthropic-key  "sk-ant-..."
caribou config set-deepseek-key   "sk-..."
```

Keys are stored in `$CARIBOU_HOME/.env` and loaded at session start.

### Step 2: Download a dataset

```bash
caribou datasets          # interactive browser + downloader
```

### Step 3: Run an agent session

```bash
# Interactive (turn-by-turn)
caribou run interactive

# Automated (fire-and-forget)
caribou run auto --turns 10 --prompt "Perform QC and generate a UMAP."
```

The run wizard prompts for blueprint, dataset, sandbox, Python analysis
environment, and LLM backend if not supplied as flags. The bundled container
environment remains the default. To use an existing host Conda, Mamba,
Micromamba, pyenv, or virtualenv prefix instead:

```bash
caribou run interactive \
  --sandbox singularity \
  --python-env /absolute/path/to/environment
```

CARIBOU mounts the selected prefix read-only at the same absolute path and uses
its `bin/python` for agent-generated analysis code. The prefix must be visible on
the machine running CARIBOU and binary-compatible with the selected container.
Docker environments must also contain `ipykernel`. Invalid or incompatible
environments fail session startup; CARIBOU never silently falls back to the
bundled interpreter. Host environments are mutable developer inputs and do not
replace frozen container images for controlled experiment or benchmark runs.

DeepSeek uses exact V4 model IDs and exposes three locked modes:

```bash
# Fast responses: DeepSeek V4 Flash with thinking disabled
caribou run auto --llm deepseek --turns 10 --prompt "Perform QC."

# Fast responses: DeepSeek V4.1 Flash with thinking disabled
caribou run auto --llm deepseek-v4.1 --turns 10 --prompt "Perform QC."

# Deliberate responses: DeepSeek V4 Pro with thinking enabled at high effort
caribou run auto --llm deepseek-thinking --turns 10 --prompt "Perform QC."
```

The existing `deepseek` backend name remains compatible and now resolves to
`deepseek-v4-flash`; CARIBOU no longer relies on the retiring `deepseek-chat`
alias. Web session records persist the exact model ID and effective mode, and
CLI runs using `--make-report` write both `model` and `model_parameters` into
the session report.

OpenRouter is available as a first-class provider through its account-filtered
model catalogue:

```bash
caribou config set-openrouter-key "sk-or-..."
caribou config list-openrouter-models
caribou config list-openrouter-models --json
caribou run interactive --llm openrouter --model anthropic/claude-sonnet-4
```

The web UI provides searchable model selection and links to the full
[OpenRouter catalogue](https://openrouter.ai/models). Interactive sessions use
OpenRouter's normal provider routing with zero retention and provider data
collection denied. Reproducible experiments additionally freeze one explicit
upstream endpoint, disable fallback routing, and persist the canonical model,
routing policy, returned provider, token usage, and provider-reported cost.

---

## Web Frontend

The web UI provides a full browser-based interface to CARIBOU running on HPC.

| Session list | New session wizard |
|---|---|
| ![Session list](docs/images/web-ui-sessions.png) | ![New session wizard](docs/images/web-ui-new-session.png) |

### Build the frontend

```bash
cd frontend
npm install
npm run build          # outputs to caribou/src/caribou/frontend/browser/
```

### Start the server

```bash
caribou serve                    # binds to 0.0.0.0:8000
caribou serve --port 9000        # custom port
caribou serve --refresh          # backend reload + Angular browser refresh
```

At startup, `caribou serve` displays the experiment access token used by the
guided experiment interface. If no token is configured, CARIBOU generates one,
saves it in the protected CARIBOU environment file, and reuses it on later
starts. Retrieve the same token at any time with:

```bash
caribou config get-control-token
```

Treat this value as a deployment secret: anyone who has it can operate durable
experiments on that CARIBOU server. It is not a model-provider API key.
The browser sends it in the proxy-safe `X-Caribou-Control-Token` header so it
does not collide with Open OnDemand's own `Authorization` handling. Direct API
clients may continue using `Authorization: Bearer <token>` for compatibility.

FastAPI serves both the REST/WebSocket API and the Angular static bundle. The server auto-detects the Open OnDemand node-proxy path pattern and strips it transparently — no extra flags needed.

### Access from your browser

**Open OnDemand (OOD):**
```
https://<ood-host>/node/<compute-node>/<port>/
# e.g. https://ondemand.mskcc.org/node/iscb014/8000/
```

**SSH tunnel (any HPC with SSH access):**
```bash
ssh -L 8000:localhost:8000 user@login-node
# then open http://localhost:8000
```

**VS Code Remote / VS Code Server:**
Open the Ports panel (bottom panel → Ports tab), click **Forward a Port**, enter `8000`. VS Code provides a proxied URL.

### Development (hot reload)

Let `caribou serve` start both the backend and Angular dev server:

```bash
caribou serve --refresh
```

`--refresh` starts Uvicorn with backend reload and starts Angular on port `4200`
with browser auto-refresh. Use `--frontend-port 4201` if port `4200` is already
in use. FastAPI remains available on the backend port, usually `8000`.

### Frontend features

| Feature | Details |
|---|---|
| **Real-time streaming** | Token-by-token LLM output with live cursor |
| **Markdown rendering** | Full GFM: code blocks with syntax highlighting, bold, lists, headers |
| **Plot previews** | Inline thumbnails in the sidebar; click to open full-resolution lightbox |
| **Session persistence** | Sessions survive server restarts; message history, artifacts, and event log are saved to disk |
| **Error visibility** | Errors shown inline in the chat stream and in a sidebar error log with timestamps |
| **Status timeline** | Per-turn status log showing agent name, turn number, and state transitions |
| **Auto mode** | Sessions with an initial prompt start running immediately after sandbox init — no user action needed |
| **Delete confirmation** | Destructive actions require explicit confirmation |
| **Settings page** | Configure sessions directory and API keys from the browser |

### Session storage

Sessions are persisted to `$CARIBOU_HOME/server_sessions/<session-id>/`:

```
server_sessions/
  <uuid>/
    session.json      ← metadata, messages, event log
    outputs/          ← plots, .h5ad files, CSVs from the sandbox
```

The sessions directory can be changed from the web UI Settings page or by setting `CARIBOU_SESSIONS_DIR` in `$CARIBOU_HOME/.env`.

---

## CLI Command Reference

### `caribou run`

```bash
caribou run interactive                          # guided interactive session
caribou run auto --turns 10 --prompt "..."       # automated N-turn run
caribou run auto \
  --blueprint ~/.local/share/caribou/agent_systems/my_system.json \
  --dataset   ~/.local/share/caribou/datasets/my_data.h5ad \
  --sandbox   singularity \
  --python-env /absolute/path/to/environment \
  --llm       claude \
  --turns     20 \
  --output-dir ./results/run_001
```

### `caribou serve`

```bash
caribou serve                     # start web server on 0.0.0.0:8000
caribou serve --port 9000         # custom port
caribou serve --reload            # backend Python auto-reload only
caribou serve --refresh           # backend reload + Angular browser refresh
caribou serve --refresh --frontend-port 4201
```

### `caribou create-system`

```bash
caribou create-system             # interactive agent blueprint builder
caribou create-system quick --name my-system
```

### `caribou datasets`

```bash
caribou datasets                  # interactive browser
caribou datasets download --version stable --dataset-id "<uuid>"
```

### `caribou config`

```bash
caribou config set-openai-key     "sk-..."
caribou config set-anthropic-key  "sk-ant-..."
caribou config set-deepseek-key   "sk-..."
```

---

## Configuration

CARIBOU stores all user data in `CARIBOU_HOME`. Override the location with the environment variable.

| Platform | Default path |
|---|---|
| Linux | `~/.local/share/caribou/` |
| macOS | `~/Library/Application Support/caribou/` |
| Windows | `%LOCALAPPDATA%\OpenTechBio\caribou\` |

| Path | Contents |
|---|---|
| `$CARIBOU_HOME/.env` | API keys |
| `$CARIBOU_HOME/agent_systems/` | Custom agent blueprints (JSON) |
| `$CARIBOU_HOME/datasets/` | Downloaded datasets (.h5ad) |
| `$CARIBOU_HOME/server_sessions/` | Web UI session history and outputs |
| `$CARIBOU_HOME/server_uploads/` | Datasets uploaded via the web UI |
| `$CARIBOU_HOME/runs/` | CLI run outputs and chat logs |

---

## Architecture

```
Browser
  └── Angular SPA (frontend/)
        ├── HTTP  →  FastAPI REST  (caribou/server/)
        └── WS    →  FastAPI WebSocket
                       └── session_manager.py
                             └── streaming_runner.py
                                   ├── AgentSystem  (agents/)
                                   ├── LLM backend  (core/)
                                   └── Sandbox      (sandbox/)
                                         └── Jupyter kernel (in container)
```

The web server layer is entirely separate from the existing CLI (`caribou run`). Both drive the same underlying agent execution engine.

---

CARIBOU builds on the original [OLAF](https://github.com/OpenTechBio/Olaf) project, adapted for high-performance, reproducible execution on GPU and HPC environments.
