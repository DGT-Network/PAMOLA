"""pamola_core/cli/commands/schema.py"""

import json
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table

from pamola_core.utils.ops.op_registry import (
    discover_operations,
    get_operation_class,
    get_operation_metadata,
    get_operation_version,
)
from pamola_core.cli.utils.exit_codes import EXIT_ERROR

console = Console()


class SchemaFormat(str, Enum):
    pretty = "pretty"
    json = "json"


def show_schema(
    operation: str = typer.Argument(
        ..., help="Operation class name, e.g. AggregateRecordsOperation."
    ),
    fmt: SchemaFormat = typer.Option(
        SchemaFormat.pretty,
        "--format",
        "-f",
        help="Output format: pretty (default) or json.",
    ),
):
    """
    Display the parameter schema for an operation.

    Examples:

      pamola-core schema AggregateRecordsOperation

      pamola-core schema AggregateRecordsOperation --format json
    """
    discover_operations("pamola_core")
    op_cls = get_operation_class(operation)
    if op_cls is None:
        console.print(f"[red]✗ Unknown operation:[/red] {operation}")
        console.print(
            "  Run [bold]pamola-core list-ops[/bold] to see available operations."
        )
        raise typer.Exit(EXIT_ERROR)

    meta = get_operation_metadata(operation) or {}
    version = get_operation_version(operation) or "—"
    params = meta.get("parameters", {})

    if fmt == SchemaFormat.json:
        schema = {
            "operation": operation,
            "version": version,
            "module": meta.get("module", "—"),
            "category": meta.get("category", "general"),
            "parameters": {
                name: {
                    "type": info.get("annotation"),
                    "required": info.get("is_required", False),
                    "default": info.get("default"),
                }
                for name, info in params.items()
            },
        }
        typer.echo(json.dumps(schema, indent=2, default=str))
    else:
        _render_table(operation, version, meta, params)


def _render_table(op_name, version, meta, params):
    console.print(
        f"\n[bold cyan]{op_name}[/bold cyan]  "
        f"[dim]v{version}[/dim]  "
        f"[magenta]{meta.get('category', 'general')}[/magenta]"
    )
    console.print(f"[dim]{meta.get('module', '')}[/dim]\n")

    if not params:
        console.print("[yellow]No parameters defined.[/yellow]")
        return

    t = Table(show_lines=True, highlight=True)
    t.add_column("Parameter", style="cyan", no_wrap=True)
    t.add_column("Type", style="yellow")
    t.add_column("Required", style="red", justify="center")
    t.add_column("Default", style="green")

    for name, info in params.items():
        required = "[red]✓[/red]" if info.get("is_required") else ""
        default = "" if info.get("is_required") else str(info.get("default", ""))
        t.add_row(name, info.get("annotation") or "—", required, default)

    console.print(t)
