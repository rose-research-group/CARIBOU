"""
One-shot QC execution: Single API call with all instructions.
No agent framework, memory management, or multi-turn conversation.
"""
import os
import sys
import json
import time
import argparse
import subprocess
import importlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Reuse existing CARIBOU infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "caribou" / "src"))
from caribou.core.io_helpers import extract_python_code
from caribou.core.sandbox_management import init_singularity_exec
from caribou.execution.benchmark_runner import run_benchmark
from caribou.config import ENV_FILE

class MockConsole:
    def print(self, *args, **kwargs):
        print(*args)

class OneShotRunner:
    def __init__(self, llm_backend: str, sandbox_backend: str = "singularity"):
        load_dotenv(dotenv_path=ENV_FILE)
        self.llm_backend = llm_backend
        self.llm_client, self.model_name = self._init_llm(llm_backend)
        self.sandbox_backend = sandbox_backend

    def _init_llm(self, backend: str):
        from openai import OpenAI
        if backend == "chatgpt":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), "gpt-4o"
        elif backend == "claude":
            from caribou.core.anthropic_wrapper import AnthropicClient
            return AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY")), "claude-3-sonnet-20240229"
        elif backend == "deepseek":
            return OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                         base_url="https://api.deepseek.com"), "deepseek-chat"
        raise ValueError(f"Unknown LLM backend: {backend}")

    def _init_sandbox(self, dataset_path: Path, output_dir: Path):
        if self.sandbox_backend == "singularity":
            SingularityBackend, _, _, _, _ = init_singularity_exec(
                script_dir=str(Path(__file__).parent),
                sanbox_data_path=None,
                subprocess=subprocess,
                console=MockConsole(),
                force_refresh=False
            )

            sandbox_manager = SingularityBackend()
            resources = [(dataset_path.resolve(), "/workspace/dataset.h5ad")]
            sandbox_manager.set_data(all_resources=resources, host_output_path=output_dir.resolve())
            return sandbox_manager
        else:
            raise ValueError(f"Unsupported sandbox backend: {self.sandbox_backend}")

    def _load_prompt(self, module_name: str, var_name: str) -> str:
        module = importlib.import_module(module_name)
        prompt = getattr(module, var_name, None)
        if not prompt:
            raise ValueError(f"Prompt variable {var_name} not found in module {module_name}.")
        return prompt

    def run(
        self,
        dataset_path: Path,
        output_dir: Path,
        prompt_module: str,
        prompt_var: str,
        benchmark_module: Path | None,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        # Build the single prompt
        system_prompt = """You are an expert bioinformatician specializing in single-cell RNA-seq analysis.
You will be given a data processing task. Generate complete, executable Python code to accomplish the task.
Use scanpy, scrublet, numpy, pandas, and matplotlib as needed.
Wrap all code in ```python ... ``` blocks."""

        user_prompt = self._load_prompt(prompt_module, prompt_var)

        # Make single API call
        print(f"Making one-shot API call to {self.model_name}...")
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )

        llm_response = response.choices[0].message.content
        api_time = time.time() - start_time

        # Save the response
        (output_dir / "llm_response.md").write_text(llm_response)

        # Extract and execute code
        code = extract_python_code(llm_response)
        if not code:
            return {"success": False, "error": "No code extracted from response"}

        (output_dir / "extracted_code.py").write_text(code)

        # Initialize sandbox and execute
        exec_start = time.time()
        sandbox_manager = self._init_sandbox(dataset_path, output_dir)
        
        print("Starting sandbox container...")
        if not sandbox_manager.start_container():
            return {"success": False, "error": "Failed to start sandbox container"}
        print("Sandbox container started.")

        try:
            print("Executing code in sandbox...")
            result = sandbox_manager.exec_code(code, timeout=600)
            exec_time = time.time() - exec_start
            print("Code execution finished.")
            if benchmark_module:
                run_benchmark(
                    MockConsole(),
                    sandbox_manager,
                    benchmark_module,
                    is_auto=True,
                    output_dir=output_dir,
                    metadata={"name": dataset_path.name},
                    agent_name="one_shot",
                    code_snippet=code,
                )
        finally:
            print("Stopping sandbox container...")
            sandbox_manager.stop_container()
            print("Sandbox container stopped.")

        total_time = time.time() - start_time

        # Collect results
        code_exec_attempts = 1
        code_exec_failures = 0 if result.get("status") == "ok" else 1

        return {
            "success": result.get("status") == "ok",
            "mode": "one_shot",
            "llm_backend": self.llm_backend,
            "model_name": self.model_name,
            "api_time_seconds": api_time,
            "exec_time_seconds": exec_time,
            "total_time_seconds": total_time,
            "num_api_calls": 1,
            "code_exec_attempts": code_exec_attempts,
            "code_exec_failures": code_exec_failures,
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "output_files": list(output_dir.glob("outputs/*"))
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--llm", required=True, choices=["chatgpt", "claude", "deepseek"])
    parser.add_argument("--sandbox", default="singularity", choices=["singularity", "docker"])
    parser.add_argument("--prompt-module", default="qc_prompt")
    parser.add_argument("--prompt-var", default="QC_PROMPT")
    parser.add_argument("--benchmark-module", type=Path, default=None)
    args = parser.parse_args()

    runner = OneShotRunner(args.llm, args.sandbox)
    result = runner.run(
        args.dataset,
        args.output_dir,
        args.prompt_module,
        args.prompt_var,
        args.benchmark_module,
    )

    # Save metrics
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
