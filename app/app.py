import logging

import typer
from rich.logging import RichHandler
from typing_extensions import Annotated

from src.opensynth.datasets.low_carbon_london import get_data

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

app = typer.Typer(context_settings=dict(max_content_width=800))


@app.command()
def get_lcl_data(
    download: Annotated[
        bool, typer.Option("--download", help="Downloads LCL data.")
    ] = False,
    split: Annotated[
        bool,
        typer.Option(
            "--split", help="Splits LCL households into training/ holdout set"
        ),
    ] = False,
    preprocess: Annotated[
        bool,
        typer.Option(
            "--preprocess",
            help="Preprocesses LCL data to create 48-half hour daily"
            "load profiles",
        ),
    ] = False,
):
    """
    Download, split and preprocess the Low Carbon London dataset.
    """
    get_data.get_lcl_data(
        download=download, split=split, preprocess=preprocess
    )


if __name__ == "__main__":
    app()
