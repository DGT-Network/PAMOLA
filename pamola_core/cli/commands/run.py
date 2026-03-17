"""
pamola_core/cli/commands/run.py
REQ: FR-EP3-CORE-023 — execute task from JSON / single operation
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from pamola_core.utils.tasks.task_runner import TaskRunner
from pamola_core.utils.ops.op_registry import discover_operations
from pamola_core.cli.utils.exit_codes import EXIT_ERROR, EXIT_VALIDATION

app = typer.Typer(help="Execute a task or single operation.")
console = Console()


@app.callback(invoke_without_command=True)
def run(
    # ── Task-mode ──────────────────────────────────────────────────────────
    task: Optional[Path] = typer.Option(
        None,
        "--task",
        "-t",
        help="Path to task JSON definition file.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for task results.",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducible results.",
    ),
    # ── Single-Operation mode ─────────────────────────────────────────────────────
    op: Optional[str] = typer.Option(
        None,
        "--op",
        help="Operation class name (single-op mode), e.g. AttributeSuppressionOperation.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to operation config JSON (single-op mode).",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    input_data: Optional[Path] = typer.Option(
        None,
        "--input",
        help="Path to input CSV/Parquet (single-op mode).",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
):
    """
    Execute a task pipeline or a single operation.

    [bold]Task-mode (full pipeline):[/bold]

      pamola-core run --task task.json --output ./results

      pamola-core run --task task.json --output ./results --seed 42

    [bold]Single-Operation mode:[/bold]

      pamola-core run --op AttributeSuppressionOperation --config cfg.json --input data.csv
    """
    if task:
        _run_task(task, output, seed)
    elif op:
        _run_single_op(op, config, input_data, output, seed)
    else:
        console.print("[red]✗ Provide either --task or --op.[/red]")
        console.print("  Run [bold]pamola-core run --help[/bold] for usage.")
        raise typer.Exit(EXIT_ERROR)


# ─────────────────────────────────────────────────────────────────────────
# Task-mode
# ─────────────────────────────────────────────────────────────────────────


def _run_task(task_path: Path, output: Optional[Path], seed: Optional[int]):
    # 1. Parse JSON
    try:
        task_def = json.loads(task_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[red]✗ Invalid JSON in task file:[/red] {e}")
        raise typer.Exit(EXIT_VALIDATION)

    # 2. Extract TaskRunner parameters
    task_id = task_def.get("task_id", task_path.stem)
    task_type = task_def.get("task_type", "anonymization")
    description = task_def.get("description", "")
    input_datasets = task_def.get("input_datasets", {})
    auxiliary_datasets = task_def.get("auxiliary_datasets", {})
    data_types = task_def.get("data_types", {})
    operation_configs = task_def.get("operations", [])

    additional_options = task_def.get("additional_options", {})
    if seed is not None:
        additional_options["seed"] = seed

    # 3. Validate required fields
    if not input_datasets:
        console.print("[red]✗ Task JSON must define 'input_datasets'.[/red]")
        raise typer.Exit(EXIT_VALIDATION)

    if not operation_configs:
        console.print("[yellow]⚠ Task has no operations defined.[/yellow]")
        raise typer.Exit(EXIT_VALIDATION)

    # If output is a plain name (no path separators), place it under ./output/
    if output:
        p = Path(output)
        if p == Path(p.name):  # no directory component
            p = Path("output") / p
        task_dir = str(p)
    else:
        task_dir = None

    console.print(
        f"\n[bold]Task:[/bold] {task_path.name}  |  "
        f"[bold]Operations:[/bold] {len(operation_configs)}  |  "
        f"[bold]Output:[/bold] {task_dir or 'auto'}\n"
    )

    # 4. Run via TaskRunner (discover ops first so registry is populated)
    discover_operations("pamola_core")
    try:
        runner = TaskRunner(
            task_id=task_id,
            task_type=task_type,
            description=description,
            input_datasets=input_datasets,
            auxiliary_datasets=auxiliary_datasets,
            operation_configs=operation_configs,
            additional_options=additional_options,
            data_types=data_types,
            task_dir=task_dir,
        )
        success = runner.run()
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]✗ Task failed:[/red] {e}")
        raise typer.Exit(EXIT_ERROR)

    if success:
        console.print("\n[green]✓ Task completed successfully.[/green]")
    else:
        console.print("\n[red]✗ Task completed with errors. Check logs for details.[/red]")
        raise typer.Exit(EXIT_ERROR)


# ─────────────────────────────────────────────────────────────────────────
# Single-Operation mode
# ─────────────────────────────────────────────────────────────────────────


def _run_single_op(
    op_name: str,
    config_path: Optional[Path],
    input_path: Optional[Path],
    output: Optional[Path],
    seed: Optional[int],
):
    if not input_path:
        console.print("[red]✗ --input is required for single-op mode.[/red]")
        raise typer.Exit(EXIT_ERROR)

    # Load operation parameters from config file
    op_kwargs = {}
    if config_path:
        try:
            op_kwargs = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Invalid config JSON:[/red] {e}")
            raise typer.Exit(EXIT_VALIDATION)

    # Support structured config format: {"operation":..., "parameters":{...}, "scope":{...}}
    # or flat format: {"param1": val, "scope": {...}}
    if "parameters" in op_kwargs:
        scope = op_kwargs.pop("scope", {"target": []})
        op_kwargs.pop("operation", None)
        op_kwargs = op_kwargs.pop("parameters", {})
    else:
        scope = op_kwargs.pop("scope", {"target": []})
        op_kwargs.pop("operation", None)

    if seed is not None:
        op_kwargs["seed"] = seed

    input_name = input_path.stem
    task_dir = str(output) if output else f"./output/single_{op_name.lower()}"

    console.print(
        f"\n[bold]Single-Operation:[/bold] {op_name}  |  [bold]Input:[/bold] {input_path}\n"
    )

    # Wrap in a minimal TaskRunner execution
    discover_operations("pamola_core")
    try:
        runner = TaskRunner(
            task_id=f"single_{op_name.lower()}",
            task_type="anonymization",
            description=f"Single operation: {op_name}",
            input_datasets={input_name: str(input_path)},
            auxiliary_datasets={},
            operation_configs=[
                {
                    "operation": "single_op",
                    "class_name": op_name,
                    "parameters": op_kwargs,
                    "scope": scope,
                    "dataset_name": input_name,
                    "task_operation_id": "op_001",
                    "task_operation_order_index": 1,
                }
            ],
            additional_options={},
            task_dir=task_dir,
        )
        success = runner.run()
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[red]✗ Operation failed:[/red] {e}")
        raise typer.Exit(EXIT_ERROR)

    if success:
        console.print("\n[green]✓ Operation completed successfully.[/green]")
    else:
        console.print("\n[red]✗ Operation failed. Check logs for details.[/red]")
        raise typer.Exit(EXIT_ERROR)
