import glob as glob_module
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress

from gcphotom.surveys.snls import (
    default_output_path,
    filter_to_reference,
    parse_snls_catalog,
    process_single,
    write_catalog,
)

app = typer.Typer(help="gcphotom – Growth Curve Photometry CLI")
snls_app = typer.Typer(help="SNLS survey forced-photometry catalog processing.")
app.add_typer(snls_app, name="snls")
console = Console()


def _expand_glob(pattern):
    files = sorted(glob_module.glob(pattern))
    if not files:
        console.print(f"[red]No files matching:[/red] {pattern}")
        raise typer.Exit(1)
    return files


@snls_app.command()
def process(
    catalog: str = typer.Argument(
        ..., help="Glob pattern or path to forced photometry catalog(s)."
    ),
    reference: Path = typer.Option(
        ..., "--reference", "-r", help="Reference catalog (.npy)."
    ),
    band: str = typer.Option(
        "g", "--band", "-b", help="Reference band for star selection."
    ),
    min_flux: float = typer.Option(
        10000.0, "--min-flux", help="Minimum reference flux for star selection."
    ),
    output_dir: Path = typer.Option(
        ".", "--output-dir", "-o", help="Output directory for processed catalogs."
    ),
    learning_rate: float = typer.Option(
        5e-3, "--learning-rate", help="Adam learning rate."
    ),
    niter: int = typer.Option(10000, "--niter", help="Number of optimizer iterations."),
):
    """Fit growth curves on forced photometry catalogs matched to a reference star catalog."""
    ref_cat = np.load(reference)
    fnames = _expand_glob(catalog)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(fnames))
        for fname in fnames:
            progress.update(task, description=f"Processing {Path(fname).name}")

            result = process_single(
                fname,
                ref_cat,
                band=band,
                min_flux=min_flux,
                learning_rate=learning_rate,
                niter=niter,
            )

            if result is None:
                console.print(
                    f"  [yellow]No matched sources in[/yellow] {Path(fname).name}"
                )
                progress.advance(task)
                continue

            out_path = output_dir / default_output_path(fname).name
            write_catalog(result, out_path)
            console.print(
                f"  [green]{result['n_selected']}/{result['n_initial']}[/green]"
                f" sources written to [cyan]{out_path.name}[/cyan]"
            )
            progress.advance(task)


@snls_app.command()
def match(
    catalog: str = typer.Argument(
        ..., help="Glob pattern or path to forced photometry catalog(s)."
    ),
    reference: Path = typer.Option(
        ..., "--reference", "-r", help="Reference catalog (.npy)."
    ),
    band: str = typer.Option(
        "g", "--band", "-b", help="Reference band for star selection."
    ),
    min_flux: float = typer.Option(
        10000.0, "--min-flux", help="Minimum reference flux for star selection."
    ),
):
    """Show matching statistics between forced catalogs and a reference catalog."""
    ref_cat = np.load(reference)
    fnames = _expand_glob(catalog)

    console.print(f"{'File':60s} {'Total':>8s} {'Matched':>8s} {'Frac':>8s}")
    console.print("-" * 84)
    for fname in fnames:
        _, meta = parse_snls_catalog(fname)
        cat = meta["cat"]
        mask = filter_to_reference(cat, ref_cat, band=band, min_flux=min_flux)
        n_total = len(cat)
        n_matched = int(mask.sum())
        frac = f"{n_matched / n_total:.1%}" if n_total > 0 else "N/A"
        console.print(f"{Path(fname).name:60s} {n_total:8d} {n_matched:8d} {frac:>8s}")
