"""
pamola_core/cli/commands/list_ops.py
FR-EP3-CORE-022: list all operations from operations_catalog.yaml (NFR-EP3-CORE-120).
Falls back to runtime registry scan if catalog unavailable.
"""

import json
from enum import Enum
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

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
        help="Filter by category (e.g. profiling, anonymization, transformations, metrics, attacks, fake_data).",
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

    Reads from operations_catalog.yaml (NFR-EP3-CORE-120).

    Examples:

      pamola-core list-ops

      pamola-core list-ops --category profiling

      pamola-core list-ops --format json
    """
    try:
        ops_data = _load_from_catalog(category)
    except Exception as e:
        console.print(f"[red]✗ Failed to load operations:[/red] {e}")
        raise typer.Exit(EXIT_ERROR)

    if not ops_data:
        msg = (
            f"No operations found in category [cyan]{category}[/cyan]."
            if category
            else "No operations registered."
        )
        console.print(f"[yellow]{msg}[/yellow]")
        raise typer.Exit(0)

    if fmt == OutputFormat.json:
        typer.echo(json.dumps(ops_data, indent=2))
    else:
        _render_table(ops_data, category)

    if fmt == OutputFormat.table and not category:
        cats = sorted({o["category"] for o in ops_data})
        console.print(
            f"\n[dim]Categories: {', '.join(cats)}"
            f"  |  Use --category <name> to filter[/dim]"
        )


def _load_from_catalog(category: Optional[str]) -> list:
    """Load operations from catalog; fall back to runtime registry scan if needed."""
    try:
        from pamola_core.catalogs import get_operations_catalog
        entries = get_operations_catalog()
        ops_data = [
            {
                "name": e.get("name", ""),
                "category": e.get("category", "general"),
                "module": e.get("module", "—"),
                "version": e.get("version", "—"),
                "description": e.get("description", ""),
            }
            for e in entries
        ]
    except Exception:
        # Fallback: runtime registry scan
        from pamola_core.utils.ops.op_registry import (
            discover_operations,
            list_operations,
            get_operation_class,
            get_operation_metadata,
            get_operation_version,
        )
        discover_operations("pamola_core")
        op_names = list_operations(category=category)
        ops_data = []
        for name in sorted(op_names):
            meta = get_operation_metadata(name) or {}
            op_cls = get_operation_class(name)
            raw_doc = (op_cls.__doc__ or "") if op_cls else ""
            description = raw_doc.strip().split("\n")[0].strip()
            ops_data.append({
                "name": name,
                "category": meta.get("category", "general"),
                "module": meta.get("module", "—"),
                "version": get_operation_version(name) or "—",
                "description": description,
            })
        if category:
            ops_data = [o for o in ops_data if o["category"] == category]
        return ops_data

    if category:
        ops_data = [o for o in ops_data if o["category"] == category]

    return sorted(ops_data, key=lambda o: (o["category"], o["name"]))


def _render_table(ops_data: list, category: Optional[str]):
    title = f"PAMOLA Operations — {category}" if category else "PAMOLA Operations"
    t = Table(title=title, show_lines=True, highlight=True)
    t.add_column("Operation", style="cyan", no_wrap=True)
    t.add_column("Category", style="magenta", justify="center")
    t.add_column("Version", style="green", justify="center")
    t.add_column("Description", style="dim")

    for o in ops_data:
        t.add_row(o["name"], o["category"], o["version"], o.get("description", ""))

    console.print(t)
    console.print(f"[dim]Total: {len(ops_data)} operation(s)[/dim]")
