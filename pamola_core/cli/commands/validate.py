"""pamola_core/cli/commands/validate.py"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from pamola_core.utils.ops.op_registry import (
    get_operation_class,
    get_operation_metadata,
)
from pamola_core.cli.utils.exit_codes import EXIT_ERROR, EXIT_VALIDATION

app = typer.Typer(help="Validate operation config or task JSON files.")
console = Console()


@app.callback(invoke_without_command=True)
def validate_config(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to single-operation config JSON.",
        exists=True,
        readable=True,
    ),
    task: Optional[Path] = typer.Option(
        None,
        "--task",
        help="Path to task pipeline JSON.",
        exists=True,
        readable=True,
    ),
):
    """
    Validate a configuration or task JSON file.

    Examples:

      pamola-core validate-config --config config.json

      pamola-core validate-config --task task.json
    """
    if not config and not task:
        console.print("[red]✗ Provide --config or --task.[/red]")
        raise typer.Exit(EXIT_ERROR)

    target = config or task

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[red]✗ Invalid JSON in[/red] [cyan]{target}[/cyan]\n  {e}")
        raise typer.Exit(EXIT_VALIDATION)

    errors = _validate_task(data) if task else _validate_op_config(data)

    if errors:
        console.print(
            f"[red]✗ Validation failed[/red] — {len(errors)} issue(s) in [cyan]{target}[/cyan]\n"
        )
        for err in errors:
            console.print(f"  [red]•[/red] {err}")
        raise typer.Exit(EXIT_VALIDATION)

    console.print(f"[green]✓ Valid:[/green] [cyan]{target}[/cyan]")


def _validate_task(data: dict) -> list:
    errors = []
    if not isinstance(data, dict):
        return ["Root must be a JSON object."]
    steps = data.get("steps")
    if not steps:
        errors.append("Missing 'steps' key or empty steps list.")
        return errors
    for i, step in enumerate(steps, 1):
        op_name = step.get("operation")
        if not op_name:
            errors.append(f"Step {i}: missing 'operation' key.")
            continue
        op_cls = get_operation_class(op_name)
        if op_cls is None:
            errors.append(f"Step {i}: operation '{op_name}' not found in registry.")
            continue
        meta = get_operation_metadata(op_name) or {}
        known_params = set(meta.get("parameters", {}).keys())
        unknown = set(step.get("config", {}).keys()) - known_params
        if unknown:
            errors.append(
                f"Step {i} ({op_name}): unknown config key(s): {', '.join(sorted(unknown))}."
            )
        for param, info in meta.get("parameters", {}).items():
            if info.get("is_required") and param not in step.get("config", {}):
                errors.append(
                    f"Step {i} ({op_name}): required parameter '{param}' is missing."
                )
    return errors


def _validate_op_config(data: dict) -> list:
    errors = []
    op_name = data.get("operation")
    if not op_name:
        errors.append("Missing 'operation' key in config.")
        return errors
    op_cls = get_operation_class(op_name)
    if op_cls is None:
        errors.append(f"Operation '{op_name}' not found in registry.")
        return errors
    meta = get_operation_metadata(op_name) or {}
    for param, info in meta.get("parameters", {}).items():
        if info.get("is_required") and param not in data.get("config", {}):
            errors.append(f"Required parameter '{param}' is missing.")
    return errors
