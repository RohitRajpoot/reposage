import typer
from rich import print
from pathlib import Path

# Import and alias the stub so it doesn't collide with our CLI function
from .chat import chat as chat_plugin
from .heatmap import show_heatmap

app = typer.Typer(help="RepoSage Synth CLI")

@app.command()
def hello(name: str = "world"):
    """Warm-up command: prints Hello, {name}!"""
    print(f"[bold green]Hello, {name}![/]")

@app.command()
def heatmap():
    """Show token-similarity heatmap from tensor.pt."""
    show_heatmap()

@app.command()
def chat(repo: Path = Path(".")):
    """Invoke the chat plugin stub."""
    # Call the aliased stub, not this function
    response = chat_plugin(repo)
    print(response)

if __name__ == "__main__":
    app()