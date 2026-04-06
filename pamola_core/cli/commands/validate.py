"""pamola_core/cli/commands/validate.py"""

import inspect
import json
from enum import Enum
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


class OutputFormat(str, Enum):
    table = "table"
    json = "json"


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
    fmt: OutputFormat = typer.Option(
        OutputFormat.table,
        "--format",
        "-f",
        help="Output format: table (default) or json (for scripting).",
    ),
):
    """
    Validate a configuration or task JSON file.

    Examples:

      pamola-core validate-config --config config.json

      pamola-core validate-config --task task.json

      pamola-core validate-config --task task.json --format json
    """
    if not config and not task:
        if fmt == OutputFormat.json:
            _emit(fmt, {"valid": False, "errors": ["Provide --config or --task."]})
        else:
            console.print("[red]✗ Provide --config or --task.[/red]")
        raise typer.Exit(EXIT_ERROR)

    target = config or task

    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        if fmt == OutputFormat.json:
            _emit(fmt, {"valid": False, "file": str(target), "errors": [f"Invalid JSON: {e}"]})
        else:
            console.print(f"[red]✗ Invalid JSON in[/red] [cyan]{target}[/cyan]\n  {e}")
        raise typer.Exit(EXIT_VALIDATION)

    errors = _validate_task(data) if task else _validate_op_config(data)

    if errors:
        if fmt == OutputFormat.json:
            _emit(fmt, {"valid": False, "file": str(target), "errors": errors})
        else:
            console.print(
                f"[red]✗ Validation failed[/red] — {len(errors)} issue(s) in [cyan]{target}[/cyan]\n"
            )
            for err in errors:
                console.print(f"  [red]•[/red] {err}")
        raise typer.Exit(EXIT_VALIDATION)

    if fmt == OutputFormat.json:
        _emit(fmt, {"valid": True, "file": str(target), "errors": []})
    else:
        console.print(f"[green]✓ Valid:[/green] [cyan]{target}[/cyan]")


def _emit(fmt: OutputFormat, payload: dict) -> None:
    """Print result as JSON (scripting) or rich text (table/default)."""
    if fmt == OutputFormat.json:
        typer.echo(json.dumps(payload, indent=2))


def _validate_task(data: dict) -> list:
    """Validate a task pipeline JSON (FR-EP3-CORE-023: uses 'operations' + 'input_datasets' keys)."""
    errors = []
    if not isinstance(data, dict):
        return ["Root must be a JSON object."]

    if not data.get("input_datasets"):
        errors.append("Missing 'input_datasets' key or empty.")

    operations = data.get("operations")
    if not operations:
        errors.append("Missing 'operations' key or empty operations list.")
        return errors

    from pamola_core.utils.ops.op_registry import discover_operations
    discover_operations("pamola_core")

    for i, step in enumerate(operations, 1):
        op_name = step.get("class_name")
        if not op_name:
            errors.append(f"Operation {i}: missing 'class_name' key.")
            continue
        op_cls = get_operation_class(op_name)
        if op_cls is None:
            errors.append(f"Operation {i}: '{op_name}' not found in registry.")
            continue
        meta = get_operation_metadata(op_name) or {}
        params = step.get("parameters", {})
        known_params = set(meta.get("parameters", {}).keys())

        # Only check unknown params when the class does NOT absorb **kwargs
        # (i.e., all parent-class params are explicitly listed in signature)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(op_cls.__init__).parameters.values()
        )
        if not accepts_var_kwargs:
            unknown = set(params.keys()) - known_params
            if unknown:
                errors.append(
                    f"Operation {i} ({op_name}): unknown parameter(s): {', '.join(sorted(unknown))}."
                )

        # Runtime-injected params (from scope.target) — skip required check for these
        runtime_params = {"field_name"}
        for param, info in meta.get("parameters", {}).items():
            if param in runtime_params:
                continue
            if info.get("is_required") and param not in params:
                errors.append(
                    f"Operation {i} ({op_name}): required parameter '{param}' is missing."
                )
    return errors


def _validate_op_config(data: dict) -> list:
    errors = []
    op_name = data.get("operation")
    if not op_name:
        errors.append("Missing 'operation' key in config.")
        return errors

    from pamola_core.utils.ops.op_registry import discover_operations
    discover_operations("pamola_core")

    op_cls = get_operation_class(op_name)
    if op_cls is None:
        errors.append(f"Operation '{op_name}' not found in registry.")
        return errors

    meta = get_operation_metadata(op_name) or {}
    params = data.get("parameters", {})
    scope_targets = data.get("scope", {}).get("target", [])

    # Runtime-injected params: skip if scope.target is provided
    runtime_params = {"field_name"} if scope_targets else set()

    # Unknown params check
    known_params = set(meta.get("parameters", {}).keys())
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(op_cls.__init__).parameters.values()
    )
    if not accepts_var_kwargs:
        unknown = set(params.keys()) - known_params
        if unknown:
            errors.append(f"Unknown parameter(s): {', '.join(sorted(unknown))}.")

    # Required params check
    for param, info in meta.get("parameters", {}).items():
        if param in runtime_params:
            continue
        if info.get("is_required") and param not in params:
            errors.append(f"Required parameter '{param}' is missing.")
    return errors
