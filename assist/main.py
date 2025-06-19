import typer
from rich import print
from .heatmap import show_heatmap  # we’ll create this next

app = typer.Typer(help="RepoSage Synth CLI")

@app.command()
def hello(name: str = "world"):
    """Say hello to someone."""
    print(f"[bold green]Hello, {name}![/]")

@app.command()
def heatmap(tensor_path: str = "tensor.pt"):
    """Show token-similarity heatmap."""
    show_heatmap(tensor_path)

if __name__ == "__main__":
    app()

