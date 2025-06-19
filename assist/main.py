import typer
from rich import print
from .heatmap import show_heatmap
from .chat import chat as chat_plugin

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
def chat(question: str = typer.Argument(..., help="Question to ask RepoSage")):
    """Invoke the chat plugin stub."""
    response = chat_plugin(question)
    print(response)

if __name__ == "__main__":
    app()
