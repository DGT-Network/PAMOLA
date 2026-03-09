"""pamola_core/cli/commands/list_ops.py"""

import json
from enum import Enum
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pamola_core.utils.ops.op_registry import (
    discover_operations,
    list_operations,
    list_categories,
    get_operation_metadata,
    get_operation_version,
)
from pamola_core.cli.utils.exit_codes import EXIT_ERROR

app = typer.Typer(help="List all available PAMOLA operations.")
console = Console()


class OutputFormat(str, Enum):
    table = "table"
    json = "json"


@app.callback(invoke_without_command=True)
def list_ops(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (e.g. profiling, anonymization, field, dataframe).",
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        "-f",
        help="Output format: table (default) or json.",
    ),
):
    """
    List all registered PAMOLA operations.

    Examples:

      pamola-core list-ops

      pamola-core list-ops --category profiling

      pamola-core list-ops --format json
    """
    try:
        discover_operations("pamola_core")
        op_names = list_operations(category=category)
    except Exception as e:
        console.print(f"[red]✗ Failed to load registry:[/red] {e}")
        raise typer.Exit(EXIT_ERROR)

    if not op_names:
        msg = (
            f"No operations found in category [cyan]{category}[/cyan]."
            if category
            else "No operations registered."
        )
        console.print(f"[yellow]{msg}[/yellow]")
        raise typer.Exit(0)

    ops_data = []
    for name in sorted(op_names):
        meta = get_operation_metadata(name) or {}
        ver = get_operation_version(name) or "—"
        ops_data.append(
            {
                "name": name,
                "category": meta.get("category", "general"),
                "module": meta.get("module", "—"),
                "version": ver,
                "params": len(meta.get("parameters", {})),
            }
        )

    if fmt == OutputFormat.json:
        typer.echo(json.dumps(ops_data, indent=2))
    else:
        _render_table(ops_data, category)

    if fmt == OutputFormat.table and not category:
        cats = list_categories()
        console.print(
            f"\n[dim]Categories: {', '.join(cats)}"
            f"  |  Use --category <name> to filter[/dim]"
        )


def _render_table(ops_data: list, category: Optional[str]):
    title = f"PAMOLA Operations — {category}" if category else "PAMOLA Operations"
    t = Table(title=title, show_lines=True, highlight=True)
    t.add_column("Operation", style="cyan", no_wrap=True)
    t.add_column("Category", style="magenta", justify="center")
    t.add_column("Version", style="green", justify="center")
    t.add_column("# Params", style="yellow", justify="right")
    t.add_column("Module", style="dim")

    for o in ops_data:
        t.add_row(o["name"], o["category"], o["version"], str(o["params"]), o["module"])

    console.print(t)
    console.print(f"[dim]Total: {len(ops_data)} operation(s)[/dim]")
