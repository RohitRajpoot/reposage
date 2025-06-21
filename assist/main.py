import typer
from rich import print
from .chat import chat as chat_plugin

app = typer.Typer(help="RepoSage Synth CLI")

@app.command()
def hello(name: str = "world"):
    """Warm-up command: prints Hello, {name}!"""
    print(f"[bold green]Hello, {name}![/]")

@app.command()
def heatmap():
    """Show token-similarity heatmap from tensor.pt."""
    # import here so chat() doesn’t drag in sklearn
    from .heatmap import show_heatmap
    show_heatmap()

@app.command()
def chat(question: str = typer.Argument(..., help="Question to ask RepoSage")):
    """Invoke the embedding Q&A."""
    response = chat_plugin(question)
    print(response)

@app.command()
def bayes(question: str = typer.Argument(..., help="Question for Bayesian chat")):
    """Invoke the Bayesian-embedding chat."""
    from .bayes_chat import bayes_chat
    print(bayes_chat(question))

if __name__ == "__main__":
    app()
