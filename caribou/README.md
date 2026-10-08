https://github.com/OpenTechBio/CARIBOU

# CARIBOU CLI: The Open-source Language Agent Framework 🚀

**The CARIBOU CLI is a powerful command-line interface for building, testing, and running sandboxed, multi-agent AI systems.** 

It provides a robust framework for orchestrating multiple language agents that can collaborate to perform complex tasks, such as data analysis, in a secure and isolated environment.

At its core, CARIBOU allows you to define a team of specialized AI agents in a simple JSON "blueprint." You can then deploy this team into a secure sandbox (powered by Docker or Singularity) with a specific dataset and give them a high-level task to solve.

## Key Features

  * **Multi-Agent Blueprints:** Define agents, their specialized prompts, and how they delegate tasks to each other using a simple JSON configuration.
  * **Secure Sandboxing:** Execute agent-generated code in an isolated environment using **Docker** or **Singularity** to protect your host system.
  * **Interactive & Automated Modes:** Run agent systems in a turn-by-turn interactive chat for debugging or in a fully automated mode for benchmarking.
  * **Data Curation:** Includes tools to browse and download single-cell datasets from the CZI CELLxGENE Census to easily test your agents.
  * **Configuration Management:** Easily manage API keys and application settings with built-in commands.
  * **User-Friendly CLI:** A guided, interactive experience helps you configure every run, with flags available to override settings for use in scripts.

## Installation

### Prerequisites

Before installing CARIBOU, you need to have the following installed and configured on your system:

1.  **Conda or Miniforge** with Python 3.10 or newer
3.  **A Sandbox Backend:**
      * **Docker:** Must be installed and the Docker daemon must be running.
      * **Singularity (Apptainer):** Must be installed on your system.

### Install from PyPI (Recommended)
Coming soon!

### Install from Source (For Developers)

Use one shared Conda control-plane environment. On HPC, place the prefix on the
designated software filesystem rather than creating a virtual environment in
each checkout or experiment directory:

```bash
git clone https://github.com/OpenTechBio/caribou
cd caribou
export CARIBOU_CONDA_PREFIX=/path/on/shared-software/caribou
export PYTHONNOUSERSITE=1
conda env update --prefix "$CARIBOU_CONDA_PREFIX" \
  --file caribou/environment.control-plane.yml
PYTHONPATH=caribou/src conda run --prefix "$CARIBOU_CONDA_PREFIX" \
  python -m caribou.cli.main --help
```

The control-plane environment is shared by the CLI, web service, and tests.
Biological code executes in the versioned Docker/Apptainer analysis image by
default; a new Conda environment is not created for each run or Slurm job.
Developer sessions may instead select an existing read-only host prefix as
described below.

Long-running software agents should use the durable, machine-readable command
contract in [docs/agent-cli.md](docs/agent-cli.md). It covers discovery,
idempotent submission, detached execution, resumable events, cancellation, and
verified artifact retrieval.

-----

## 🚀 Quick Start Guide

This guide will walk you through setting up your API key, downloading a dataset, and launching your first interactive agent session in just a few steps.

### Step 1: Configure Your API Key

First, tell CARIBOU about your OpenAI, Anthropic (Claude), DeepSeek, or OpenRouter API key. This is a one-time setup.

```bash
caribou config set-openai-key "sk-YourSecretKeyGoesHere"
```
  
or  
  
```bash
caribou config set-deepseek-key "sk-YourSecretKeyGoesHere"
```

or
  
```bash
caribou config set-anthropic-key "sk-ant-YourSecretKeyGoesHere"
```

or

```bash
caribou config set-openrouter-key "sk-or-YourSecretKeyGoesHere"
caribou config list-openrouter-models
```


Your key will be stored securely in a local `.env` file within the CARIBOU configuration directory.

### Step 2: Download a Dataset

Next, let's get some data for our agents to analyze. Run the `datasets` command to browse and download a sample dataset from the CZI CELLxGENE Census.

```bash
# This will start the interactive dataset browser
caribou datasets
```

Follow the prompts to list versions and datasets, then use the `download` command as instructed.

### Step 3: Run an Agent System\!

Now you're ready to run an agent system. The `run` command is fully interactive if you don't provide any flags. It will guide you through selecting a blueprint, a dataset, and a sandbox environment.

```bash
caribou run interactive
```

This will trigger a series of prompts:

1.  **Select Agent System Blueprint:** Choose one of the default systems (from the Package) or one you've created (from User).
2.  **Select a driver agent:** Choose which agent in the system will receive the first instruction.
3.  **Select Dataset:** Pick the dataset you downloaded in Step 2.
4.  **Choose a sandbox backend:** Select `docker` or `singularity`.
5.  **Choose a Python environment:** Keep the bundled default or select a
    discovered host Conda/Mamba/pyenv/virtualenv prefix.
6.  **Choose an LLM backend:** Select `chatgpt`, `claude`, `deepseek`,
    `deepseek-v4.1`, `deepseek-thinking`, or `ollama`. `deepseek` is the quick
    DeepSeek V4 Flash profile, `deepseek-v4.1` is the quick DeepSeek V4.1 Flash
    profile; `deepseek-thinking` is DeepSeek V4 Pro with thinking enabled.

After configuration, the session will begin, and you can start giving instructions to your agent team\!

-----

## Command Reference

CARIBOU's commands are organized into logical groups.

### `caribou run`

The main command for executing an agent system.

  * **Run interactively (recommended for manual use):**
    ```bash
    caribou run interactive
    ```
  * **Run automatically for 5 turns:**
    ```bash
    caribou run auto --turns 5 --prompt "Analyze this dataset and generate a UMAP plot."
    ```
  * **Run with all options specified (for scripting):**
    ```bash
    caribou run interactive \
      --blueprint ~/.local/share/caribou/agent_systems/my_custom_system.json \
      --driver-agent data_analyst \
      --dataset ~/.local/share/caribou/datasets/my_data.h5ad \
      --sandbox docker \
      --python-env /absolute/path/to/environment \
      --llm chatgpt
    ```

`--python-env` mounts the selected prefix read-only at the same absolute path and
uses its `bin/python` for sandbox analysis. The environment must be visible on
the CARIBOU host and compatible with the container; Docker additionally requires
`ipykernel`. The web session form provides the same discovered-environment picker
and manual path validation. This feature executes host-provided code and exposes
server-visible paths, so the current unauthenticated web server remains suitable
only for trusted/single-user deployment. CARIBOU automatically rebuilds a cached
Docker image once if it predates host-environment launcher support.

Interactive runs expose an enforced work-item ledger to agents. Agent commands
must occupy the whole message and use these forms:

```text
open_work_item "<title>" "<body>"
close_work_item <id> "<completion summary>"
list_work_items
read_work_item <id>
delegate_to_<agent> <id>
```

Operators can inspect the same ledger with `/work-items` and `/work-item <id>`,
and request bounded evaluator review with `/review-work-item <id>`. Every
transition is committed in the run's `work-items/` Git repository. Blueprints
may set `"work_item_policy": {"qc_mode": "required"}` to make close move an
item to `In review`; required mode also needs a declared `evaluator_agent`.
Without that field, close moves directly to `Done` and review is optional.

### `caribou create-system`

Tools for building new agent system blueprints.

  * **Start the interactive builder:**
    ```bash
    caribou create-system
    ```
  * **Create a minimal blueprint quickly:**
    ```bash
    caribou create-system quick --name my-first-system
    ```

### `caribou datasets`

Tools for managing datasets.

  * **Start the interactive dataset browser:**
    ```bash
    caribou datasets
    ```
  * **Download a specific dataset directly:**
    ```bash
    caribou datasets download --version stable --dataset-id "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    ```

### `caribou config`

Manage your CARIBOU configuration.

  * **Set your OpenAI API key:**
    ```bash
    caribou config set-openai-key "sk-..."
    ```
  * **Set your DeepSeek API key:**
    ```bash
    caribou config set-deepseek-key "sk-..."
    ```
  * **Set your Anthropic API key:**
    ```bash
    caribou config set-anthropic-key "sk-ant-..."
    ```

-----

## Configuration

CARIBOU stores all user-generated content and configuration in a central directory. You can override this location by setting the `CARIBOU_HOME` environment variable.

  * **Default Location:**
      * **Linux:** `~/.local/share/caribou/`
      * **macOS:** `~/Library/Application Support/caribou/`
      * **Windows:** `C:\Users\<user>\AppData\Local\OpenTechBio\caribou\`
  * **Configuration File:** API keys are stored in `$CARIBOU_HOME/.env`.
  * **Agent Systems:** Custom blueprints are saved to `$CARIBOU_HOME/agent_systems/`.
  * **Datasets:** Downloaded datasets are stored in `$CARIBOU_HOME/datasets/`.
  * **Run Outputs:** Code snippets and logs from agent runs are saved to `$CARIBOU_HOME/runs/`.
