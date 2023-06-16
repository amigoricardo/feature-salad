import typer
from .utils.load_yaml import load_yaml
from .feature_salad import FeatureSalad

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def main(
    config: str = typer.Argument(..., help='Configuration file'),
    output: str = typer.Argument(..., help='Output file (parquet, csv or json)')
) -> None:
    c = load_yaml(config)
    fs = FeatureSalad(features=c['features'], samples=c['samples'])
    fs.generate()
    if output.endswith('.parquet'):
        fs.X.to_parquet(output)
    elif output.endswith('.csv'):
        fs.X.to_csv(output)
    elif output.endswith('.json'):
        fs.X.to_json(output)