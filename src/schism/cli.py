"""Schism CLI - reshape how AI thinks, one slider at a time."""

from __future__ import annotations

import webbrowser
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.text import Text

app = typer.Typer(
    name="schism",
    help="Reshape how AI thinks, one slider at a time.",
    no_args_is_help=False,
)
console = Console()

presets_app = typer.Typer(help="Manage personality presets")
app.add_typer(presets_app, name="presets")


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    """Launch the Schism web UI (default command)."""
    if ctx.invoked_subcommand is None:
        serve()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind to"),
    port: int = typer.Option(6660, help="Port to bind to"),
    no_browser: bool = typer.Option(False, help="Don't auto-open browser"),
):
    """Launch the Schism web UI."""
    import uvicorn

    console.print(Panel.fit(
        "[bold]SCHISM[/bold]\n"
        f"[dim]Reshape how AI thinks, one slider at a time.[/dim]\n\n"
        f"Open [bold cyan]http://{host}:{port}[/bold cyan] in your browser",
        border_style="bright_magenta",
    ))

    if not no_browser:
        webbrowser.open(f"http://{host}:{port}")

    uvicorn.run(
        "schism.server:app",
        host=host,
        port=port,
        log_level="info",
    )


@app.command()
def steer(
    prompt: str = typer.Argument(..., help="The prompt to generate from"),
    model: str = typer.Option("gemma-2-2b", "--model", "-m", help="Model to use"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Preset to apply"),
    creativity: float = typer.Option(0.0, help="Creativity slider (-1 to 1)"),
    formality: float = typer.Option(0.0, help="Formality slider (-1 to 1)"),
    humor: float = typer.Option(0.0, help="Humor slider (-1 to 1)"),
    confidence: float = typer.Option(0.0, help="Confidence slider (-1 to 1)"),
    verbosity: float = typer.Option(0.0, help="Verbosity slider (-1 to 1)"),
    empathy: float = typer.Option(0.0, help="Empathy slider (-1 to 1)"),
    technical: float = typer.Option(0.0, help="Technical slider (-1 to 1)"),
    max_tokens: int = typer.Option(256, help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
):
    """Generate text with personality steering."""
    from schism.engine.generate import generate_sync
    from schism.presets.manager import get_preset

    # Build sliders from preset or flags
    sliders = {}
    if preset:
        preset_data = get_preset(preset)
        if preset_data is None:
            console.print(f"[red]Preset '{preset}' not found[/red]")
            raise typer.Exit(1)
        sliders = preset_data["sliders"]
        console.print(f"[dim]Using preset: {preset_data['name']}[/dim]")
    else:
        # Use individual slider flags
        slider_map = {
            "creativity": creativity,
            "formality": formality,
            "humor": humor,
            "confidence": confidence,
            "verbosity": verbosity,
            "empathy": empathy,
            "technical": technical,
        }
        sliders = {k: v for k, v in slider_map.items() if abs(v) > 0.01}

    if not sliders:
        console.print("[yellow]No sliders set - output will be unsteered[/yellow]")

    # Show active sliders
    if sliders:
        slider_text = " | ".join(
            f"[cyan]{k}[/cyan]={v:+.1f}" for k, v in sliders.items() if abs(v) > 0.01
        )
        console.print(f"[dim]Sliders: {slider_text}[/dim]")

    console.print(f"[dim]Model: {model} | Loading...[/dim]")

    result = generate_sync(
        model_name=model,
        prompt=prompt,
        sliders=sliders,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    console.print()
    console.print(Panel(result, title="[bold]Steered Output[/bold]", border_style="green"))


@app.command()
def compare(
    prompt: str = typer.Argument(..., help="The prompt to compare"),
    model: str = typer.Option("gemma-2-2b", "--model", "-m", help="Model to use"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Preset to apply"),
    max_tokens: int = typer.Option(256, help="Maximum tokens to generate"),
):
    """Side-by-side comparison of steered vs vanilla output."""
    from schism.engine.generate import generate_sync
    from schism.presets.manager import get_preset

    sliders = {}
    if preset:
        preset_data = get_preset(preset)
        if preset_data is None:
            console.print(f"[red]Preset '{preset}' not found[/red]")
            raise typer.Exit(1)
        sliders = preset_data["sliders"]
        console.print(f"[dim]Comparing with preset: {preset_data['name']}[/dim]")

    console.print(f"[dim]Model: {model} | Loading...[/dim]")

    # Generate both
    console.print("[dim]Generating vanilla response...[/dim]")
    vanilla = generate_sync(model, prompt, {}, max_tokens=max_tokens)

    console.print("[dim]Generating steered response...[/dim]")
    steered = generate_sync(model, prompt, sliders, max_tokens=max_tokens)

    # Display side by side
    console.print()
    console.print(Columns([
        Panel(vanilla, title="[bold]Vanilla[/bold]", border_style="blue", width=50),
        Panel(steered, title="[bold]Steered[/bold]", border_style="green", width=50),
    ]))


@app.command()
def download(
    model: str = typer.Argument(..., help="Model to download (e.g. gemma-2-2b)"),
):
    """Download a model and its SAE weights."""
    from schism.engine.loader import load_model, MODEL_REGISTRY

    if model not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model: {model}[/red]")
        console.print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
        raise typer.Exit(1)

    console.print(f"[bold]Downloading {model}...[/bold]")
    console.print("[dim]This may take a while on first run.[/dim]")

    status = console.status(f"Loading {model}...")
    status.start()

    def on_progress(stage, detail):
        status.update(f"[{stage}] {detail}")

    try:
        load_model(model, on_progress=on_progress)
        status.stop()
        console.print(f"[green]Model {model} is ready![/green]")
    except RuntimeError as e:
        status.stop()
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def warmup(
    model: str = typer.Argument(..., help="Model to warm up (e.g. gemma-2-2b)"),
):
    """Pre-compute and cache all steering vectors for a model.

    This makes the first generation fast by computing all default
    steering vectors upfront.
    """
    from schism.engine.loader import load_model, MODEL_REGISTRY
    from schism.engine.features import get_all_features
    from schism.engine.generate import _get_or_compute_vector

    if model not in MODEL_REGISTRY:
        console.print(f"[red]Unknown model: {model}[/red]")
        console.print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
        raise typer.Exit(1)

    console.print(f"[bold]Warming up {model}...[/bold]")

    status = console.status(f"Loading {model}...")
    status.start()

    try:
        model_data = load_model(model, on_progress=lambda s, d: status.update(f"[{s}] {d}"))
    except RuntimeError as e:
        status.stop()
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    status.stop()

    features = get_all_features()
    total = len(features)

    for i, name in enumerate(sorted(features.keys()), 1):
        console.print(f"  [{i}/{total}] Computing steering vector for [cyan]{name}[/cyan]...")
        _get_or_compute_vector(model, name, model_data)

    console.print(f"\n[green]All {total} steering vectors cached for {model}![/green]")
    console.print("[dim]Subsequent generations will be much faster.[/dim]")


# --- Preset subcommands ---


@presets_app.command("list")
def presets_list():
    """List all available presets."""
    from schism.presets.manager import list_presets

    presets = list_presets()
    if not presets:
        console.print("[dim]No presets found[/dim]")
        return

    table = Table(title="Personality Presets")
    table.add_column("Name", style="bold cyan")
    table.add_column("Description")
    table.add_column("Source", style="dim")
    table.add_column("Sliders")

    for p in presets:
        active = [f"{k}={v:+.1f}" for k, v in p["sliders"].items() if abs(v) > 0.01]
        table.add_row(
            p["name"],
            p.get("description", ""),
            p.get("source", "user"),
            ", ".join(active[:4]) + ("..." if len(active) > 4 else ""),
        )

    console.print(table)


@presets_app.command("export")
def presets_export(
    name: str = typer.Argument(..., help="Preset name"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export a preset to a JSON file for sharing."""
    import json
    from schism.presets.manager import get_preset

    preset = get_preset(name)
    if not preset:
        console.print(f"[red]Preset '{name}' not found[/red]")
        raise typer.Exit(1)

    # Strip internal fields
    export_data = {
        "name": preset["name"],
        "description": preset.get("description", ""),
        "model": preset.get("model", "any"),
        "sliders": preset["sliders"],
        "version": preset.get("version", 1),
    }

    if output is None:
        output = f"{preset['name'].lower().replace(' ', '_')}.json"

    with open(output, "w") as f:
        json.dump(export_data, f, indent=2)

    console.print(f"[green]Exported to {output}[/green]")


@app.command()
def models():
    """List all available models and their status."""
    from schism.engine.loader import list_available_models

    table = Table(title="Available Models")
    table.add_column("Name", style="bold cyan")
    table.add_column("HuggingFace ID", style="dim")
    table.add_column("SAE", justify="center")
    table.add_column("Status", justify="center")

    for m in list_available_models():
        sae_status = "[green]yes[/green]" if m["has_sae"] else "[dim]no (contrastive)[/dim]"
        load_status = "[green]loaded[/green]" if m["loaded"] else "[dim]not loaded[/dim]"
        table.add_row(m["name"], m["hf_id"], sae_status, load_status)

    console.print(table)
    console.print("\n[dim]Use 'schism download <model>' to download a model[/dim]")


@presets_app.command("show")
def presets_show(name: str = typer.Argument(..., help="Preset name")):
    """Show details of a specific preset."""
    from schism.presets.manager import get_preset

    preset = get_preset(name)
    if not preset:
        console.print(f"[red]Preset '{name}' not found[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold]{preset['name']}[/bold]\n"
        f"[dim]{preset.get('description', '')}[/dim]\n"
        f"Model: {preset.get('model', 'any')}\n",
        border_style="cyan",
    ))

    for feature, value in preset["sliders"].items():
        bar_width = 20
        center = bar_width // 2
        filled = int((value + 1) / 2 * bar_width)
        bar = "░" * bar_width
        bar = bar[:filled] + "█" + bar[filled + 1:]
        color = "green" if value > 0 else "red" if value < 0 else "dim"
        console.print(f"  {feature:>12s}  [{color}]{bar}[/{color}]  {value:+.1f}")


def main():
    app()
