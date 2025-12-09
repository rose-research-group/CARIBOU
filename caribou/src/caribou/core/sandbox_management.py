
import time
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import json

from caribou.sandbox.benchmarking_sandbox_management import (
    SandboxManager as _BackendManager,
    CONTAINER_NAME as _SANDBOX_HANDLE,
    IMAGE_TAG as _SANDBOX_IMAGE,  
    API_PORT_HOST as _API_PORT,
)


def init_docker(script_dir:str, subprocess, console, force_refresh:bool=False):
    # --- optional force‑refresh logic --------------------------------------
    if force_refresh:
        console.print("[yellow]Forcing Docker sandbox refresh…[/yellow]")
        # Stop & remove any running container gracefully
        subprocess.run(["docker", "rm", "-f", _SANDBOX_HANDLE], check=False)
        # Remove the sandbox image to ensure re‑pull/build
        subprocess.run(["docker", "image", "rm", "-f", _SANDBOX_IMAGE], check=False)
        console.print("[green]Docker image removed – it will be pulled/built on next start.[/green]")

    def COPY_CMD(src: str, dst: str):
        subprocess.run(["docker", "cp", src, dst], check=True)
    
    # create sandbox directory in docker 
    EXECUTE_ENDPOINT = f"http://localhost:{_API_PORT}/execute"
    STATUS_ENDPOINT = f"http://localhost:{_API_PORT}/status"

    return _BackendManager, _SANDBOX_HANDLE, COPY_CMD, EXECUTE_ENDPOINT, STATUS_ENDPOINT




def init_singularity_exec(script_dir: str, sanbox_data_path, subprocess, console, force_refresh: bool = False):
    import caribou.sandbox.benchmarking_sandbox_management_singularity as sing

    # optional force‑refresh
    if force_refresh:
        console.print("[yellow]Forcing Singularity sandbox refresh…[/yellow]")
        if sing.SIF_PATH.exists():
            sing.SIF_PATH.unlink()
            console.print(
                f"[green]Deleted {sing.SIF_PATH.name} – it will be re‑downloaded on next start.[/green]"
            )

    SIF_PATH = sing.SIF_PATH
    SING_BIN = sing.SING_BIN
    SENTINEL = "<<<EOF>>>"

    class _SingExecBackend:
        """Launch one long‑lived REPL inside the SIF and stream code to it."""

        def __init__(self):
            self._binds: List[str] = []
            self._proc = None
            self._host_output_path: Optional[Path] = None

        def set_data(self, all_resources: List[Tuple[Path, str]], host_output_path: Path):
            """Configures all necessary bind mounts, including the output directory."""
            binds = []
            for host_path, container_path in all_resources:
                binds.extend(["--bind", f"{host_path.resolve()}:{container_path}"])

            binds.extend(["--bind", f"{host_output_path.resolve()}:/workspace/outputs"])
            self._binds = binds
            self._host_output_path = host_output_path

        # ------------------------------------------------------------------
        # Container lifecycle
        # ------------------------------------------------------------------
        def start_container(self):
            if self._proc:
                return True  # already running
            if not sing.pull_sif_if_needed():
                return False

            cmd = [
                SING_BIN,
                "exec",
                "--containall",
                "--cleanenv",
                *self._binds,
                str(SIF_PATH),
                "python",
                "/opt/offline_kernel.py",
                "--repl",
            ]
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line buffered
            )
            # Wait for the REPL banner
            ready_line = self._proc.stdout.readline().strip()
            if ready_line != "__REPL_READY__":
                console.print(
                    f"[red]REPL failed to start. Got: {ready_line}[/red]"
                )
                self.stop_container()
                return False
            return True

        def stop_container(self):
            if not self._proc:
                return True
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None
            return True

        # ------------------------------------------------------------------
        # Code execution
        # ------------------------------------------------------------------
        def exec_code(self, code: str, timeout: int = 300) -> Dict:
            if not self._proc:
                raise RuntimeError("REPL not running")
            assert self._proc.stdin and self._proc.stdout

            # Send code block + sentinel
            self._proc.stdin.write(code)
            if not code.endswith("\n"):
                self._proc.stdin.write("\n")
            self._proc.stdin.write(SENTINEL + "\n")
            self._proc.stdin.flush()

            # Read exactly one JSON line
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    return {
                        "status": "timeout",
                        "stdout": "",
                        "stderr": "Execution timed out in REPL.",
                        "images": [],
                    }
                line = self._proc.stdout.readline()
                if not line:
                    continue
                line = line.strip()
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    # Non‑JSON noise; continue reading
                    continue

        # ------------------------------------------------------------------
        # Output collection helpers
        # ------------------------------------------------------------------
        def list_output_files(self) -> List[Dict]:
            """
            For Singularity, outputs live on the host; list them if available.
            """
            out_dir = self._host_output_path
            if not out_dir or not out_dir.exists():
                return []
            return [
                {"name": f.name, "size": f"{f.stat().st_size / 1e6:.2f} MB"}
                for f in out_dir.iterdir()
                if f.is_file()
            ]

        def retrieve_output_files(self, host_destination_path: Path, file_names: Optional[List[str]] = None) -> None:
            """
            For Singularity, files are already on host_output_path; this confirms location
            or copies a selected subset to another host directory if requested.
            """
            source_dir = self._host_output_path
            if not source_dir or not source_dir.exists():
                console.print("[yellow]No output directory available to retrieve from.[/yellow]")
                return

            # If the destination is the same as the source, just acknowledge
            if host_destination_path.resolve() == source_dir.resolve():
                console.print(f"[bold green]✓ Session outputs are already saved in:[/bold green] {host_destination_path}")
                return

            host_destination_path.mkdir(parents=True, exist_ok=True)
            selected = file_names or [f.name for f in source_dir.iterdir() if f.is_file()]
            for name in selected:
                src = source_dir / name
                if src.exists() and src.is_file():
                    dest = host_destination_path / name
                    dest.write_bytes(src.read_bytes())
            console.print(f"[bold green]✓ Saved selected outputs to:[/bold green] {host_destination_path}")

    _BackendManager = _SingExecBackend

    def COPY_CMD(src: str, dst: str):
        console.print("[yellow]singularity-exec mode uses bind mounts instead of docker cp.[/yellow]")
    
    return _BackendManager, None, COPY_CMD, None, None
    
    
    
