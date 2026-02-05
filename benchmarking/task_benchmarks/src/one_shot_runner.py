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
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Reuse existing CARIBOU infrastructure
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "caribou" / "src"))
from caribou.core.io_helpers import extract_python_code
from caribou.core.sandbox_management import init_singularity_exec
from caribou.execution.benchmark_runner import run_benchmark
from caribou.auto_metrics.registry import find_metric_id_by_path
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

    def _load_prompt(self, prompt_path: Path) -> str:
        if not prompt_path.exists():
            raise ValueError(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text()

    def _check_autometric_success(self, output_dir: Path):
        """Load benchmark_results.jsonl and infer autometric success.

        Returns:
            Tuple of (autometric_success: bool | None, autometric_results: dict | None)
        """
        ledger_path = output_dir / "benchmark_results.jsonl"
        if not ledger_path.exists():
            return None, None
        lines = [line for line in ledger_path.read_text().splitlines() if line.strip()]
        if not lines:
            return None, None
        try:
            record = json.loads(lines[-1])
            results = record.get("results", {})
        except json.JSONDecodeError:
            return None, None

        if not results:
            return None, None

        # Infer success based on task type (same logic as results_collector)
        if isinstance(results.get("success"), bool):
            return results.get("success"), results

        autometric_success = None
        if "doublet_score_present" in results or "predicted_doublet_present" in results:
            # For doublet task: columns must be present AND doublets must have been filtered
            columns_present = bool(
                results.get("doublet_score_present") and results.get("predicted_doublet_present")
            )
            # predicted_doublet_rate should be ~0 after filtering (doublets removed)
            # A high rate (>5%) suggests filtering didn't happen
            doublet_rate = results.get("predicted_doublet_rate")
            if doublet_rate is None:
                autometric_success = columns_present
            else:
                filtering_worked = doublet_rate < 0.05
                autometric_success = columns_present and filtering_worked
        elif "obs_columns_present" in results:
            obs_ok = all(results.get("obs_columns_present", {}).values())
            autometric_success = bool(
                obs_ok
                and results.get("counts_layer_present")
                and results.get("pca_present")
                and results.get("umap_present")
                and results.get("hvg_calculated")
            )
        elif "n_obs" in results and "n_vars" in results:
            autometric_success = bool(results.get("n_obs", 0) > 0 and results.get("n_vars", 0) > 0)

        return autometric_success, results

    def run(
        self,
        dataset_path: Path,
        output_dir: Path,
        prompt_path: Path,
        benchmark_module: Path | None,
        benchmark_id: str | None,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Emit minimal params.txt for downstream aggregations
        params_path = output_dir / "params.txt"
        params_path.write_text(
            "\n".join(
                [
                    f"LLM_BACKEND: {self.llm_backend}",
                    f"MODEL_NAME: {self.model_name}",
                    f"DATASET_PATH: {dataset_path}",
                    f"PROMPT_PATH: {prompt_path}",
                    f"BENCHMARK_MODULE: {benchmark_module}" if benchmark_module else "BENCHMARK_MODULE: ",
                    f"BENCHMARK_ID: {benchmark_id}" if benchmark_id else "BENCHMARK_ID: ",
                    f"MODE: one_shot",
                ]
            )
            + "\n"
        )
        start_time = time.time()

        # Build the single prompt
        system_prompt = """You are an expert bioinformatician specializing in single-cell RNA-seq analysis.
You will be given a data processing task. Generate complete, executable Python code to accomplish the task.
Use scanpy, scrublet, numpy, pandas, and matplotlib as needed.
Wrap all code in ```python ... ``` blocks."""

        user_prompt = self._load_prompt(prompt_path)

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

        autometric_success = None
        autometric_results = None
        try:
            print("Executing code in sandbox...")
            result = sandbox_manager.exec_code(code, timeout=600)
            exec_time = time.time() - exec_start
            print("Code execution finished.")
            selected_benchmark_id = benchmark_id
            if selected_benchmark_id is None and benchmark_module:
                selected_benchmark_id = find_metric_id_by_path(benchmark_module)
                if selected_benchmark_id is None:
                    print(f"[caribou] Unknown benchmark metric id: {benchmark_module}")
            if selected_benchmark_id:
                run_benchmark(
                    MockConsole(),
                    sandbox_manager,
                    selected_benchmark_id,
                    is_auto=True,
                    output_dir=output_dir,
                    metadata={"name": dataset_path.name},
                    agent_name="one_shot",
                    code_snippet=code,
                )
                # Check the autometric results we just wrote
                autometric_success, autometric_results = self._check_autometric_success(output_dir)
        finally:
            print("Stopping sandbox container...")
            sandbox_manager.stop_container()
            print("Sandbox container stopped.")

        total_time = time.time() - start_time

        # Collect results
        code_exec_attempts = 1
        code_execution_success = result.get("status") == "ok"
        code_exec_failures = 0 if code_execution_success else 1
        correction_count = 0

        # Success requires both code execution AND autometric validation (if available)
        # If no benchmark was run, fall back to code execution success only
        if autometric_success is not None:
            success = code_execution_success and autometric_success
        else:
            success = code_execution_success

        return {
            "success": success,
            "code_execution_success": code_execution_success,
            "autometric_success": autometric_success,
            "autometric_results": autometric_results,
            "mode": "one_shot",
            "llm_backend": self.llm_backend,
            "model_name": self.model_name,
            "api_time_seconds": api_time,
            "exec_time_seconds": exec_time,
            "total_time_seconds": total_time,
            "num_api_calls": 1,
            "code_exec_attempts": code_exec_attempts,
            "code_exec_failures": code_exec_failures,
            "correction_count": correction_count,
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
    parser.add_argument("--prompt-path", required=True, type=Path)
    parser.add_argument("--benchmark-module", type=Path, default=None)
    parser.add_argument("--benchmark-id", type=str, default=None)
    args = parser.parse_args()

    runner = OneShotRunner(args.llm, args.sandbox)
    result = runner.run(
        args.dataset,
        args.output_dir,
        args.prompt_path,
        args.benchmark_module,
        args.benchmark_id,
    )

    # Save metrics
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()
