# caribou/cli/utils_cli.py
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm

# Import from the central config to know where chat logs are stored by default
from caribou.config import CARIBOU_HOME
from caribou.core.io_helpers import split_message_by_fence

utils_app = typer.Typer(
    name="utils",
    help="Utility commands for managing CARIBOU artifacts like chat logs.",
    no_args_is_help=True
)

console = Console()
LOG_DIR = CARIBOU_HOME / "runs" / "chat_logs"


@utils_app.command("refresh-sif")
def refresh_sif() -> None:
    """
    Force re-download the Singularity SIF used by CARIBOU.
    """
    try:
        from caribou.sandbox import benchmarking_sandbox_management_singularity as sing
    except Exception as exc:
        console.print(f"[bold red]Error importing Singularity manager: {exc}[/bold red]")
        raise typer.Exit(1)

    console.print("[yellow]Refreshing Singularity sandbox SIF...[/yellow]")
    if not sing.pull_sif_if_needed(force_pull=True):
        console.print("[bold red]Failed to refresh the Singularity SIF.[/bold red]")
        raise typer.Exit(1)
    console.print("[bold green]Singularity SIF refreshed.[/bold green]")

def _convert_history_to_notebook(history_path: Path, output_path: Path):
    """
    Parses an CARIBOU chat log and converts it into a Jupyter Notebook (.ipynb).
    """
    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error: Could not read or parse the history file at {history_path}.[/bold red]\n{e}")
        raise typer.Exit(1)

    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11" # This can be made more dynamic if needed
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    for message in history:
        role = message.get("role")
        content = message.get("content", "")
        
        # We are primarily interested in the agent's responses
        if role and "assistant" in role:
            parts = split_message_by_fence(content)
            for kind, part in parts:
                if kind == "code":
                    cell = {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": part,
                    }
                else:
                    cell = {"cell_type": "markdown", "metadata": {}, "source": part}
                notebook["cells"].append(cell)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        console.print(f"[bold green]✓ Successfully converted chat log to notebook:[/bold green] {output_path}")
    except Exception as e:
        console.print(f"[bold red]Error writing notebook file: {e}[/bold red]")
        raise typer.Exit(1)

@utils_app.command("convert-to-notebook")
def convert_to_notebook(
    chat_log: Path = typer.Argument(
        ...,
        help="Path to the interactive chat log JSON file to convert.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the output .ipynb file. Defaults to the same name as the input file.",
        writable=True,
        resolve_path=True,
    ),
):
    """
    Converts an CARIBOU interactive chat log into an executable Jupyter Notebook.
    
    This command parses the JSON log file, extracts all Python code blocks generated
    by the assistant, and arranges them into code cells. The explanatory text
    between code blocks is converted into markdown cells, creating a clean,
    reproducible protocol of the agent session.
    """
    if not chat_log.name.startswith("interactive_chat_"):
        console.print(f"[yellow]Warning: The input file '{chat_log.name}' does not look like a standard CARIBOU chat log.[/yellow]")

    output_path = output
    if output_path is None:
        # Default to the same name as the input file, but with an .ipynb extension
        output_path = chat_log.with_suffix(".ipynb")
    
    # Ensure the output path has the correct extension
    if output_path.suffix != ".ipynb":
        output_path = output_path.with_suffix(".ipynb")

    console.print(f"Converting [cyan]{chat_log.name}[/cyan] to Jupyter Notebook...")
    _convert_history_to_notebook(chat_log, output_path)


def _check_singularity_available() -> bool:
    """Check if Singularity/Apptainer is available on the system."""
    sing_bin = shutil.which("apptainer") or shutil.which("singularity")
    return sing_bin is not None


def _check_docker_available() -> bool:
    """Check if Docker is available and daemon is running."""
    if not shutil.which("docker"):
        return False

    # Try to ping docker daemon
    try:
        import subprocess
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _test_sandbox_execution(sandbox_type: str) -> bool:
    """
    Launch a sandbox and test Python execution.

    Args:
        sandbox_type: Either "singularity" or "docker"

    Returns:
        True if test execution succeeded, False otherwise
    """
    console.print(f"\n[bold cyan]Launching {sandbox_type} sandbox...[/bold cyan]")

    try:
        if sandbox_type == "singularity":
            import subprocess
            from pathlib import Path
            from caribou.core.sandbox_management import init_singularity_exec

            # Use a temporary output directory for the test
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="caribou_sandbox_check_"))

            # Initialize singularity backend
            script_dir = Path(__file__).resolve().parent
            manager_class, handle, copy_cmd, exec_endpoint, status_endpoint = init_singularity_exec(
                script_dir,
                "/workspace/dataset.h5ad",  # placeholder, not used in test
                subprocess,
                console,
                force_refresh=False
            )

            sandbox_manager = manager_class()
            # Set minimal data configuration (no datasets, just output dir)
            sandbox_manager.set_data([], temp_dir)
        else:  # docker
            from caribou.sandbox.benchmarking_sandbox_management import SandboxManager
            sandbox_manager = SandboxManager()

        # Start the sandbox
        console.print("[cyan]Starting container...[/cyan]")
        sandbox_manager.start_container()
        console.print("[green]✓ Container started successfully[/green]")

        # Test 1: Simple Python execution
        test_code = "print('Hello from CARIBOU sandbox!')"
        console.print(f"\n[cyan]Test 1: Basic Python execution[/cyan]")
        console.print(f"[dim]Code: {test_code}[/dim]")

        result = sandbox_manager.exec_code(test_code, timeout=30)

        # Check result based on sandbox type
        success = False
        if sandbox_type == "singularity":
            # Singularity exec mode returns {"status": "ok"|"error", "stdout": "...", "stderr": "..."}
            if result.get("status") == "ok":
                console.print(f"[green]✓ Basic execution succeeded![/green]")
                if result.get("stdout"):
                    console.print(f"[dim]Output: {result['stdout'].strip()}[/dim]")
                success = True
            else:
                console.print(f"[red]✗ Basic execution failed[/red]")
                if result.get("stderr"):
                    console.print(f"[red]Error: {result['stderr']}[/red]")
                success = False
        else:  # docker
            # Docker API mode returns {"outputs": [...], "final_status": "ok"|"error"|"timeout"}
            if result.get("final_status") == "ok":
                console.print(f"[green]✓ Basic execution succeeded![/green]")
                outputs = result.get("outputs", [])
                for output in outputs:
                    if output.get("type") == "stream" and output.get("name") == "stdout":
                        console.print(f"[dim]Output: {output.get('text', '').strip()}[/dim]")
                success = True
            else:
                console.print(f"[red]✗ Basic execution failed[/red]")
                success = False

        # If basic test failed, don't continue
        if not success:
            console.print("\n[yellow]Skipping remaining tests due to basic execution failure[/yellow]")
            return False

        # Test 2: GPU/CUDA availability
        cuda_test_code = """import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('No CUDA devices detected')"""

        console.print(f"\n[cyan]Test 2: GPU/CUDA availability[/cyan]")
        console.print(f"[dim]Checking PyTorch CUDA support...[/dim]")

        cuda_result = sandbox_manager.exec_code(cuda_test_code, timeout=60)

        # Check CUDA test result
        cuda_available = False
        if sandbox_type == "singularity":
            if cuda_result.get("status") == "ok":
                console.print(f"[green]✓ CUDA check completed[/green]")
                if cuda_result.get("stdout"):
                    output = cuda_result['stdout'].strip()
                    console.print(f"[dim]{output}[/dim]")
                    cuda_available = "CUDA available: True" in output
                    if not cuda_available:
                        console.print(f"[yellow]⚠ Warning: CUDA not available in container despite --nv flag[/yellow]")
            else:
                console.print(f"[red]✗ CUDA check failed[/red]")
                if cuda_result.get("stderr"):
                    console.print(f"[red]Error: {cuda_result['stderr']}[/red]")
        else:  # docker
            if cuda_result.get("final_status") == "ok":
                console.print(f"[green]✓ CUDA check completed[/green]")
                outputs = cuda_result.get("outputs", [])
                for output in outputs:
                    if output.get("type") == "stream" and output.get("name") == "stdout":
                        output_text = output.get('text', '').strip()
                        console.print(f"[dim]{output_text}[/dim]")
                        cuda_available = "CUDA available: True" in output_text
                        if not cuda_available:
                            console.print(f"[yellow]⚠ Warning: CUDA not available in container[/yellow]")
            else:
                console.print(f"[red]✗ CUDA check failed[/red]")

        # Test 3: RAPIDS import
        rapids_test_code = """import rapids_singlecell as rsc
print(f'rapids-singlecell version: {rsc.__version__}')
print('RAPIDS import successful!')"""

        console.print(f"\n[cyan]Test 3: RAPIDS-singlecell import[/cyan]")
        console.print(f"[dim]Importing rapids_singlecell...[/dim]")

        rapids_result = sandbox_manager.exec_code(rapids_test_code, timeout=60)

        # Check RAPIDS test result
        if sandbox_type == "singularity":
            if rapids_result.get("status") == "ok":
                console.print(f"[green]✓ RAPIDS import succeeded![/green]")
                if rapids_result.get("stdout"):
                    console.print(f"[dim]{rapids_result['stdout'].strip()}[/dim]")
                success = True
            else:
                console.print(f"[red]✗ RAPIDS import failed[/red]")
                if rapids_result.get("stderr"):
                    console.print(f"[red]Error: {rapids_result['stderr']}[/red]")
                success = False
        else:  # docker
            if rapids_result.get("final_status") == "ok":
                console.print(f"[green]✓ RAPIDS import succeeded![/green]")
                outputs = rapids_result.get("outputs", [])
                for output in outputs:
                    if output.get("type") == "stream" and output.get("name") == "stdout":
                        console.print(f"[dim]{output.get('text', '').strip()}[/dim]")
                success = True
            else:
                console.print(f"[red]✗ RAPIDS import failed[/red]")
                success = False

        # Clean up
        console.print("\n[cyan]Stopping container...[/cyan]")
        sandbox_manager.stop_container()
        console.print("[green]✓ Container stopped[/green]")

        # Clean up temporary directory if created
        if sandbox_type == "singularity":
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        return success

    except Exception as e:
        console.print("[bold red]Error during sandbox test:[/bold red]")
        console.print(str(e), markup=False)  # Disable markup to avoid issues with brackets in error messages
        # Try to clean up
        try:
            sandbox_manager.stop_container()
        except Exception:
            pass
        # Clean up temp dir if it exists
        if sandbox_type == "singularity":
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        return False


@utils_app.command("container-rebuild")
def container_rebuild(
    container_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Container type to rebuild: 'singularity', 'docker', or 'both'",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force rebuild without confirmation",
    ),
):
    """
    Rebuild CARIBOU sandbox containers.

    This command rebuilds the sandbox containers from their definition files.

    PERMISSIONS REQUIRED:
    - Singularity: Requires --fakeroot flag OR sudo permissions
      * With fakeroot: singularity build --fakeroot (no sudo needed)
      * With sudo: sudo singularity build (requires root access)
    - Docker: Requires user to be in 'docker' group OR sudo permissions

    The rebuild is necessary after modifying:
    - environment.yml (conda packages)
    - Singularity or Dockerfile (container definition)
    - Any files copied into containers during build
    """
    console.print("[bold]CARIBOU Container Rebuild Utility[/bold]\n")

    # Ask which container to rebuild if not specified
    if not container_type:
        try:
            container_type = Prompt.ask(
                "Which container would you like to rebuild?",
                choices=["singularity", "docker", "both"],
                default="singularity"
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled by user[/yellow]")
            raise typer.Exit(0)

    container_type = container_type.lower()
    if container_type not in ["singularity", "docker", "both"]:
        console.print(f"[bold red]Invalid container type: {container_type}[/bold red]")
        console.print("Must be 'singularity', 'docker', or 'both'")
        raise typer.Exit(1)

    # Get sandbox directory
    from pathlib import Path
    sandbox_dir = Path(__file__).resolve().parent.parent / "sandbox"

    if not sandbox_dir.exists():
        console.print(f"[bold red]Sandbox directory not found: {sandbox_dir}[/bold red]")
        raise typer.Exit(1)

    # Confirm rebuild
    if not force:
        console.print(f"\n[yellow]This will rebuild the {container_type} container(s).[/yellow]")
        console.print("[yellow]This may take several minutes and will replace existing containers.[/yellow]\n")

        try:
            should_proceed = Confirm.ask("Do you want to proceed?", default=True)
            if not should_proceed:
                console.print("[yellow]Rebuild cancelled[/yellow]")
                raise typer.Exit(0)
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled by user[/yellow]")
            raise typer.Exit(0)

    import subprocess
    success = True

    # Rebuild Singularity
    if container_type in ["singularity", "both"]:
        console.print("\n[bold cyan]Rebuilding Singularity container...[/bold cyan]")
        console.print(f"[dim]Location: {sandbox_dir / 'sandbox.sif'}[/dim]")
        console.print(f"[dim]Definition: {sandbox_dir / 'Singularity'}[/dim]\n")

        # Check if singularity is available
        sing_bin = shutil.which("apptainer") or shutil.which("singularity")
        if not sing_bin:
            console.print("[bold red]✗ Singularity/Apptainer not found in PATH[/bold red]")
            success = False
        else:
            console.print(f"[dim]Using: {sing_bin}[/dim]")
            console.print("[dim]Building with --fakeroot (no sudo required)...[/dim]\n")

            try:
                result = subprocess.run(
                    [
                        sing_bin,
                        "build",
                        "--fakeroot",
                        "--force",
                        str(sandbox_dir / "sandbox.sif"),
                        str(sandbox_dir / "Singularity")
                    ],
                    cwd=sandbox_dir,
                    capture_output=False,  # Show output in real-time
                    text=True
                )

                if result.returncode == 0:
                    console.print("\n[green]✓ Singularity container rebuilt successfully[/green]")
                else:
                    console.print(f"\n[red]✗ Singularity build failed with exit code {result.returncode}[/red]")
                    console.print("\n[yellow]If --fakeroot failed, try with sudo:[/yellow]")
                    console.print(f"[dim]sudo {sing_bin} build --force sandbox.sif Singularity[/dim]")
                    success = False
            except Exception as e:
                console.print(f"\n[bold red]Error building Singularity container:[/bold red]")
                console.print(str(e), markup=False)
                success = False

    # Rebuild Docker
    if container_type in ["docker", "both"]:
        console.print("\n[bold cyan]Rebuilding Docker container...[/bold cyan]")
        console.print(f"[dim]Location: {sandbox_dir}[/dim]")
        console.print(f"[dim]Dockerfile: {sandbox_dir / 'Dockerfile'}[/dim]\n")

        # Check if docker is available
        if not shutil.which("docker"):
            console.print("[bold red]✗ Docker not found in PATH[/bold red]")
            success = False
        else:
            console.print("[dim]Building Docker image...[/dim]\n")

            try:
                result = subprocess.run(
                    [
                        "docker",
                        "build",
                        "--no-cache",
                        "-t",
                        "caribou-sandbox:latest",
                        "."
                    ],
                    cwd=sandbox_dir,
                    capture_output=False,  # Show output in real-time
                    text=True
                )

                if result.returncode == 0:
                    console.print("\n[green]✓ Docker container rebuilt successfully[/green]")
                else:
                    console.print(f"\n[red]✗ Docker build failed with exit code {result.returncode}[/red]")
                    console.print("\n[yellow]If permission denied, ensure you're in the docker group:[/yellow]")
                    console.print("[dim]sudo usermod -aG docker $USER[/dim]")
                    console.print("[dim]Then log out and back in[/dim]")
                    success = False
            except Exception as e:
                console.print(f"\n[bold red]Error building Docker container:[/bold red]")
                console.print(str(e), markup=False)
                success = False

    # Final status
    console.print("\n" + "="*60)
    if success:
        console.print("[bold green]✓ Container rebuild completed successfully[/bold green]")
        console.print("\nYou can now run 'caribou utils sandbox-check' to verify the containers.")
    else:
        console.print("[bold red]✗ Container rebuild failed[/bold red]")
        console.print("\nPlease check the errors above and try again.")
        raise typer.Exit(1)
    console.print("="*60)


@utils_app.command("sandbox-check")
def sandbox_check():
    """
    Check sandbox availability and test execution.

    This command helps verify that your sandbox environment is properly configured.
    It will:
    1. Ask which sandbox type you want to check (Singularity or Docker)
    2. Check if the sandbox is available
    3. Offer to download container if needed (Singularity only)
    4. Launch the sandbox and execute a simple Python test
    5. Report results
    """
    console.print("[bold]CARIBOU Sandbox Check Utility[/bold]\n")

    # Ask which sandbox to check
    try:
        sandbox_choice = Prompt.ask(
            "Which sandbox would you like to check?",
            choices=["singularity", "docker"],
            default="singularity"
        )
        sandbox_type = sandbox_choice.lower()
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Checking {sandbox_type.capitalize()} sandbox...[/bold]\n")

    # Check availability
    if sandbox_type == "singularity":
        if not _check_singularity_available():
            console.print("[bold red]✗ Singularity/Apptainer is not installed or not in PATH[/bold red]")
            console.print("\nPlease install Singularity or Apptainer:")
            console.print("  - Apptainer: https://apptainer.org/docs/admin/main/installation.html")
            console.print("  - Singularity: https://sylabs.io/guides/latest/user-guide/")
            raise typer.Exit(1)

        console.print("[green]✓ Singularity/Apptainer is available[/green]")

        # Check if container exists
        from caribou.sandbox.benchmarking_sandbox_management_singularity import (
            SIF_PATH,
            pull_sif_if_needed
        )

        if SIF_PATH.exists():
            console.print(f"[green]✓ Container found at:[/green] {SIF_PATH}")
        else:
            console.print(f"[yellow]Container not found at:[/yellow] {SIF_PATH}")

            # Ask if user wants to download
            try:
                should_download = Confirm.ask(
                    "Would you like to download the Singularity container now?",
                    default=True
                )
                if not should_download:
                    console.print("[yellow]Skipping download. Cannot proceed with sandbox test.[/yellow]")
                    raise typer.Exit(0)
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled by user[/yellow]")
                raise typer.Exit(0)

            # Download the container
            console.print("\n[bold cyan]Downloading Singularity container...[/bold cyan]")
            console.print("[dim]This may take several minutes...[/dim]\n")

            try:
                success = pull_sif_if_needed(force_pull=False)
                if success:
                    console.print("[green]✓ Container downloaded successfully[/green]")
                else:
                    console.print("[bold red]✗ Failed to download container[/bold red]")
                    raise typer.Exit(1)
            except Exception as e:
                console.print("[bold red]Error downloading container:[/bold red]")
                console.print(str(e), markup=False)
                raise typer.Exit(1)

    else:  # docker
        if not _check_docker_available():
            console.print("[bold red]✗ Docker is not installed or daemon is not running[/bold red]")
            console.print("\nPlease install Docker and ensure the daemon is running:")
            console.print("  - Docker: https://docs.docker.com/get-docker/")
            raise typer.Exit(1)

        console.print("[green]✓ Docker is available and daemon is running[/green]")

        # Check if Docker image exists, if not it will be built automatically
        console.print("[dim]Docker image will be built automatically if needed[/dim]")

    # Test execution
    console.print(f"\n[bold]Testing {sandbox_type.capitalize()} execution...[/bold]")
    success = _test_sandbox_execution(sandbox_type)

    # Final report
    console.print("\n" + "="*60)
    if success:
        console.print(f"[bold green]✓ {sandbox_type.capitalize()} sandbox check PASSED[/bold green]")
        console.print(f"\nYour {sandbox_type} sandbox is ready to use!")
    else:
        console.print(f"[bold red]✗ {sandbox_type.capitalize()} sandbox check FAILED[/bold red]")
        console.print(f"\nPlease check the errors above and try again.")
        raise typer.Exit(1)
    console.print("="*60)
