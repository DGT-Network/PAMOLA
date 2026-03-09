"""
pamola_core/cli/main.py
Root CLI entry point for PAMOLA.CORE
Entry point: `pamola-core` (defined in pyproject.toml)
"""

import logging
from typing import Optional

import typer

app = typer.Typer(
    name="pamola-core",
    help="[bold cyan]PAMOLA.CORE[/bold cyan] — Privacy-Preserving AI Data Processing Framework",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)



def _version_callback(value: bool):
    if value:
        from pamola_core import __version__  # adjust if your version lives elsewhere

        typer.echo(f"pamola-core {__version__}")
        raise typer.Exit(0)


def _verbose_callback(value: bool):
    if value:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            force=True,
        )


@app.callback()
def _root(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    verbose: Optional[bool] = typer.Option(
        None,
        "--verbose",
        "-v",
        help="Enable debug logging.",
        callback=_verbose_callback,
        is_eager=True,
        expose_value=False,
    ),
):
    """PAMOLA.CORE CLI — Privacy-Preserving Data Processing Framework."""


# ── Register sub-commands ──────────────────────────────────────────────────
from pamola_core.cli.commands.list_ops import app as list_ops_app  # noqa: E402
from pamola_core.cli.commands.run import app as run_app  # noqa: E402
from pamola_core.cli.commands.validate import app as validate_app  # noqa: E402
from pamola_core.cli.commands.schema import app as schema_app  # noqa: E402

app.add_typer(list_ops_app, name="list-ops")
app.add_typer(run_app, name="run")
app.add_typer(validate_app, name="validate-config")
app.add_typer(schema_app, name="schema")

if __name__ == "__main__":
    app()
