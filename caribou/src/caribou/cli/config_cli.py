# caribou/cli/config_cli.py
import re
import typer
from rich.console import Console
from typing import Optional

# Import the centralized ENV_FILE path
from caribou.config import ENV_FILE

config_app = typer.Typer(
    name="config",
    help="Manage CARIBOU configuration and API keys.",
    no_args_is_help=True
)

console = Console()

@config_app.command("set-openai-key")
def set_api_key(
    api_key: str = typer.Argument(..., help="Your OpenAI API key (e.g., 'sk-...')")
):
    """
    Saves your OpenAI API key to the CARIBOU environment file.
    """
    if not api_key.startswith("sk-"):
        console.print("[yellow]Warning: Key does not look like a standard OpenAI API key (should start with 'sk-').[/yellow]")

    # Ensure the .env file exists
    if not ENV_FILE.exists():
        ENV_FILE.touch()

    content = ENV_FILE.read_text()
    key_to_set = f'OPENAI_API_KEY="{api_key}"'

    # Use regex to safely replace the key if it already exists
    if re.search(r"^OPENAI_API_KEY=.*$", content, flags=re.MULTILINE):
        new_content = re.sub(r"^OPENAI_API_KEY=.*$", key_to_set, content, flags=re.MULTILINE)
    else:
        new_content = content + f"\n{key_to_set}\n"

    ENV_FILE.write_text(new_content.strip())
    console.print(f"[bold green]✅ OpenAI API key has been set successfully in:[/bold green] {ENV_FILE}")

@config_app.command("set-deepseek-key")
def set_deepseek_key(
    ctx: typer.Context,
    api_key: Optional[str] = typer.Argument(None, help="Your DeepSeek API key (e.g., 'ds_...')"),
):
    """
    Saves your DeepSeek API key to the Caribou environment file.
    """
    if api_key is None:
        console.print("[bold red]Error:[/bold red] You must provide an API key.\n")
        typer.echo(ctx.parent.get_help())
        raise typer.Exit()

    if not api_key.startswith("ds_"):
        console.print(
            "[yellow]Warning: Key does not look like a standard DeepSeek API key (should start with 'ds_').[/yellow]"
        )

    if not ENV_FILE.exists():
        ENV_FILE.touch()

    content = ENV_FILE.read_text()
    key_to_set = f'DEEPSEEK_API_KEY="{api_key}"'

    if re.search(r"^DEEPSEEK_API_KEY=.*$", content, flags=re.MULTILINE):
        new_content = re.sub(r"^DEEPSEEK_API_KEY=.*$", key_to_set, content, flags=re.MULTILINE)
    else:
        new_content = content.strip() + f"\n{key_to_set}\n"

    ENV_FILE.write_text(new_content)
    console.print(f"[bold green]✅ DeepSeek API key has been set successfully in:[/bold green] {ENV_FILE}")