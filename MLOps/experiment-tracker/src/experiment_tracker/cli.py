import typer
from tabulate import tabulate
from database import ExperimentDB

app = typer.Typer(help="A streamlined CLI for tracking ML experiments")
db = ExperimentDB("tracker_test.db")

@app.command(name="list-exp")
def list_experiments():
    """
    Show a list of all existing experiments
    """
    experiments = db.get_all_experiments()
    if not experiments:
        typer.echo("No experiments found yet. Database is empty.")
        return
        
    table_data = []
    for exp in experiments:
        table_data.append([exp["id"], exp["name"], exp["created_at"]])
        
    typer.echo(typer.style("\n List of ML Experiments", fg=typer.colors.CYAN, bold=True))
    typer.echo(tabulate(table_data, headers=["ID", "Experiment title", "Date of creation"], tablefmt="grid"))
    typer.echo("")

@app.command(name="show")
def show_experiment_results(
    experiment_name: str = typer.Argument(..., help="Experiment name to display")
):
    """
    Show a summary table of results for a specific experiment
    """
    summary = db.get_experiment_runs_summary(experiment_name)
    if not summary:
        typer.echo(f"Experiment with a name '{experiment_name}' not found or there are no launches in it.")
        return
        
    typer.echo(typer.style(f"\nResults of the experiment: {experiment_name}", fg=typer.colors.GREEN, bold=True))
    typer.echo(tabulate(summary, headers="keys", tablefmt="fancy_grid", missingval="-"))
    typer.echo("")

if __name__ == "__main__":
    app()


'''
launch examples:

python experiment_tracker/cli.py --help

python experiment_tracker/cli.py list-exp

python experiment_tracker/cli.py show "Context Manager Test"
'''