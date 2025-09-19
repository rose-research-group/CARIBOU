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

1.  **Python** (version 3.9 or higher)
2.  **Pip** (Python's package installer)
3.  **A Sandbox Backend:**
      * **Docker:** Must be installed and the Docker daemon must be running.
      * **Singularity (Apptainer):** Must be installed on your system.

### Install from PyPI (Recommended)
Coming soon!

### Install from Source (For Developers)

To install the latest development version, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/OpenTechBio/caribou
cd caribou/cli/caribou
pip install -e .
caribou
```

-----

## 🚀 Quick Start Guide

This guide will walk you through setting up your API key, downloading a dataset, and launching your first interactive agent session in just a few steps.

### Step 1: Configure Your API Key

First, tell CARIBOU about your OpenAI or DeepSeek API key. This is a one-time setup.

```bash
caribou config set-openai-key "sk-YourSecretKeyGoesHere"
```
  
or  
  
```bash
caribou config set-deepseek-key "sk-YourSecretKeyGoesHere"
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
5.  **Choose an LLM backend:** Select `chatgpt` or `ollama`.

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
      --llm chatgpt
    ```

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