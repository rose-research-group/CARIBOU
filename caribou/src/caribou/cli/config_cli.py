# caribou/cli/config_cli.py
import os
import re
import json
import typer
from rich.console import Console
from typing import Optional

# Import the centralized ENV_FILE path
from caribou.config import (
    ENV_FILE,
    InvalidSlurmPartitionError,
    get_caribou_slurm_partition,
    validate_slurm_partition,
)
from caribou.core.control_access import resolve_control_access_token
from dotenv import load_dotenv, set_key

config_app = typer.Typer(
    name="config",
    help="Manage CARIBOU configuration and API keys.",
    no_args_is_help=True,
)

console = Console()


def _save_provider_key(name: str, value: str) -> None:
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch(mode=0o600)
    set_key(str(ENV_FILE), name, value)
    os.chmod(ENV_FILE, 0o600)


@config_app.command("get-control-token")
def get_control_token(
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Print only the token, for copying or command substitution.",
    ),
) -> None:
    """Show the persistent token used by the experiment web interface."""

    resolved = resolve_control_access_token(create=True, env_file=ENV_FILE)
    if resolved is None:  # Defensive: create=True always returns a token.
        raise typer.Exit(1)
    if raw:
        typer.echo(resolved.value)
        return

    source = {
        "environment": "CARIBOU_CONTROL_API_TOKEN in the current environment",
        "caribou_env_file": f"CARIBOU environment file: {resolved.env_file}",
        "generated": f"new token saved in: {resolved.env_file}",
    }[resolved.source]
    console.print("[bold]Experiment access token[/bold]")
    console.print(resolved.value, markup=False)
    console.print(f"[dim]Source: {source}[/dim]")
    console.print(
        "[yellow]Anyone with this token can operate experiments on this CARIBOU server. "
        "Do not paste a model-provider API key here or share this output publicly.[/yellow]"
    )


@config_app.command("set-slurm-partition")
def set_slurm_partition(
    partition: str = typer.Argument(
        ..., help="Slurm partition CARIBOU should submit and bind jobs on."
    ),
) -> None:
    """Set the Slurm partition used for every CARIBOU Slurm execution."""
    try:
        validate_slurm_partition(partition)
    except InvalidSlurmPartitionError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    _save_provider_key("CARIBOU_SLURM_PARTITION", partition)
    console.print(
        f"[bold green]✅ Slurm partition set to '{partition}' in:[/bold green] {ENV_FILE}"
    )
    console.print(
        "[dim]Runs already bound to the previous partition remain valid; "
        "only new submissions use the updated partition. Restart 'caribou serve' "
        "if it is already running.[/dim]"
    )


@config_app.command("show-slurm-partition")
def show_slurm_partition() -> None:
    """Show the Slurm partition CARIBOU is currently configured to use."""
    console.print(get_caribou_slurm_partition(), markup=False)


@config_app.command("set-openai-key")
def set_api_key(
    api_key: str = typer.Argument(..., help="Your OpenAI API key (e.g., 'sk-...')"),
):
    """
    Saves your OpenAI API key to the CARIBOU environment file.
    """
    if not api_key.startswith("sk-"):
        console.print(
            "[yellow]Warning: Key does not look like a standard OpenAI API key (should start with 'sk-').[/yellow]"
        )

    # Ensure the .env file exists
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch()
    os.chmod(ENV_FILE, 0o600)

    content = ENV_FILE.read_text()
    key_to_set = f'OPENAI_API_KEY="{api_key}"'

    # Use regex to safely replace the key if it already exists
    if re.search(r"^OPENAI_API_KEY=.*$", content, flags=re.MULTILINE):
        new_content = re.sub(
            r"^OPENAI_API_KEY=.*$", key_to_set, content, flags=re.MULTILINE
        )
    else:
        new_content = content + f"\n{key_to_set}\n"

    ENV_FILE.write_text(new_content.strip())
    os.chmod(ENV_FILE, 0o600)
    console.print(
        f"[bold green]✅ OpenAI API key has been set successfully in:[/bold green] {ENV_FILE}"
    )


@config_app.command("set-deepseek-key")
def set_deepseek_key(
    ctx: typer.Context,
    api_key: Optional[str] = typer.Argument(
        None, help="Your DeepSeek API key (e.g., 'ds_...')"
    ),
):
    """
    Saves your DeepSeek API key to the Caribou environment file.
    """
    if api_key is None:
        console.print("[bold red]Error:[/bold red] You must provide an API key.\n")
        if ctx.parent is not None:
            typer.echo(ctx.parent.get_help())
        raise typer.Exit()

    if not api_key.startswith("ds_"):
        console.print(
            "[yellow]Warning: Key does not look like a standard DeepSeek API key (should start with 'ds_').[/yellow]"
        )

    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch()
    os.chmod(ENV_FILE, 0o600)

    content = ENV_FILE.read_text()
    key_to_set = f'DEEPSEEK_API_KEY="{api_key}"'

    if re.search(r"^DEEPSEEK_API_KEY=.*$", content, flags=re.MULTILINE):
        new_content = re.sub(
            r"^DEEPSEEK_API_KEY=.*$", key_to_set, content, flags=re.MULTILINE
        )
    else:
        new_content = content.strip() + f"\n{key_to_set}\n"

    ENV_FILE.write_text(new_content)
    os.chmod(ENV_FILE, 0o600)
    console.print(
        f"[bold green]✅ DeepSeek API key has been set successfully in:[/bold green] {ENV_FILE}"
    )


@config_app.command("set-anthropic-key")
def set_anthropic_key(
    ctx: typer.Context,
    api_key: Optional[str] = typer.Argument(
        None, help="Your Anthropic API key (e.g., 'sk-ant-...')"
    ),
):
    """
    Saves your Anthropic API key to the Caribou environment file.
    """
    if api_key is None:
        console.print("[bold red]Error:[/bold red] You must provide an API key.\n")
        if ctx.parent is not None:
            typer.echo(ctx.parent.get_help())
        raise typer.Exit()

    if not api_key.startswith("sk-"):
        console.print(
            "[yellow]Warning: Key does not look like a standard Anthropic API key (should start with 'sk-').[/yellow]"
        )

    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not ENV_FILE.exists():
        ENV_FILE.touch()
    os.chmod(ENV_FILE, 0o600)

    content = ENV_FILE.read_text()
    key_to_set = f'ANTHROPIC_API_KEY="{api_key}"'

    if re.search(r"^ANTHROPIC_API_KEY=.*$", content, flags=re.MULTILINE):
        new_content = re.sub(
            r"^ANTHROPIC_API_KEY=.*$", key_to_set, content, flags=re.MULTILINE
        )
    else:
        new_content = content.strip() + f"\n{key_to_set}\n"

    ENV_FILE.write_text(new_content.strip())
    os.chmod(ENV_FILE, 0o600)
    console.print(
        f"[bold green]✅ Anthropic API key has been set successfully in:[/bold green] {ENV_FILE}"
    )


@config_app.command("set-openrouter-key")
def set_openrouter_key(
    api_key: str = typer.Argument(..., help="Your OpenRouter API key (sk-or-...)"),
) -> None:
    """Save the OpenRouter inference API key in CARIBOU's private environment file."""

    if not api_key.startswith("sk-or-"):
        console.print(
            "[yellow]Warning: key does not look like a standard OpenRouter key "
            "(expected sk-or-...).[/yellow]"
        )
    _save_provider_key("OPENROUTER_API_KEY", api_key)
    console.print(
        f"[bold green]✅ OpenRouter API key saved in:[/bold green] {ENV_FILE}"
    )


@config_app.command("list-openrouter-models")
def list_openrouter_models(
    json_output: bool = typer.Option(
        False, "--json", help="Print one machine-readable JSON object."
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Bypass the five-minute cache."
    ),
) -> None:
    """List models allowed by the configured OpenRouter account."""

    from caribou.core.openrouter import OpenRouterError, get_openrouter_catalogue

    load_dotenv(dotenv_path=ENV_FILE, override=False)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        console.print("[red]OPENROUTER_API_KEY is not configured.[/red]", stderr=True)
        raise typer.Exit(1)
    try:
        catalogue = get_openrouter_catalogue(key, refresh=refresh)
    except OpenRouterError as exc:
        console.print(f"[red]{exc}[/red]", stderr=True)
        raise typer.Exit(1) from exc
    payload = catalogue.as_dict()
    if json_output:
        typer.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    stale = " [stale cache]" if catalogue.stale else ""
    console.print(f"[bold]OpenRouter models[/bold]{stale}")
    for model in catalogue.models:
        console.print(f"{model.canonical_slug}\t{model.name}", markup=False)
