import logging

import typer
from rich.logging import RichHandler
from typing_extensions import Annotated

from opensynth.datasets.low_carbon_london import get_data
from opensynth.datasets.new_england import get_data as ne_get_data
from opensynth.datasets.new_england import preprocess_ne, recs, sampling

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

app = typer.Typer(context_settings=dict(max_content_width=800))


@app.command()
def download_lcl_data(
    data_dir: Annotated[
        str, typer.Option("--loc", help="Downloads LCL data to <location>.")
    ] = "./data"
):
    """
    Download the Low Carbon London dataset.
    """
    get_data.download_lcl_data(data_dir)


@app.command()
def get_ne_data(
    data_dir: Annotated[
        str, typer.Option("--loc", help="Location of data directory.")
    ] = "./data",
    download_timeseries: Annotated[
        bool,
        typer.Option(
            "--timeseries",
            help="Also build the building manifest and download "
            "per-building EULP timeseries (~10 GB).",
        ),
    ] = False,
):
    """
    Download New England source data: EULP metadata, RECS 2020
    microdata and GHCN-Daily temperatures. With --timeseries, also
    select the stratified training buildings and download their
    15-minute profiles.
    """
    from pathlib import Path

    ne_get_data.get_ne_data(data_dir)
    if download_timeseries:
        metadata = ne_get_data.load_eulp_metadata(data_dir)
        df_recs = recs.load_recs(
            Path(data_dir) / "raw/new_england/recs/recs2020_public_v7.csv"
        )
        manifest = sampling.select_buildings(
            metadata, recs.joint_distribution(df_recs)
        )
        sampling.write_manifest(manifest, data_dir)
        ne_get_data.download_eulp_timeseries(manifest, data_dir)


@app.command()
def preprocess_ne_data(
    data_dir: Annotated[
        str, typer.Option("--loc", help="Location of data directory.")
    ] = "./data",
    pv_shape_path: Annotated[
        str,
        typer.Option(
            "--pv_shapes",
            help="Path to the PVWatts shape CSV. If omitted, PV "
            "augmentation is skipped.",
        ),
    ] = "",
    sample_fraction: Annotated[
        float,
        typer.Option(
            "--sample_fraction",
            help="Fraction of households in the training set.",
        ),
    ] = 0.75,
):
    """
    Preprocess downloaded EULP buildings into packed daily profiles
    with New England conditioning labels and DER augmentation.
    """
    from pathlib import Path

    preprocess_ne.preprocess_ne_data(
        data_dir,
        pv_shape_path=Path(pv_shape_path) if pv_shape_path else None,
        sample_fraction=sample_fraction,
    )


@app.command()
def preprocess_data(
    split: Annotated[
        bool,
        typer.Option(
            "--split", help="Splits LCL households into training/holdout set"
        ),
    ] = False,
    preprocess: Annotated[
        bool,
        typer.Option(
            "--preprocess",
            help="Preprocesses LCL data into daily load profiles",
        ),
    ] = False,
    data_dir: Annotated[
        str, typer.Option("--loc", help="Location of data directory.")
    ] = "./data",
    csv_data_path: Annotated[
        str,
        typer.Option(
            "--csv_path",
            help="Path to dataset CSV file containing, relative to data_dir.",
        ),
    ] = "raw/CC_LCL-FullData.csv",
    sample_fraction: Annotated[
        float,
        typer.Option(
            "--sample_fraction",
            help="Fraction of households to include in the training set. \
                Remaining fraction assigned to the holdout set. \
                Value between 0 and 1.",
        ),
    ] = 0.75,
    time_resolution: Annotated[
        str,
        typer.Option(
            "--time_resolution",
            help='Time resolution of the data, either "half_hourly" or \
                "hourly".',
        ),
    ] = "half_hourly",
    feature_cols: Annotated[
        list[str],
        typer.Option(
            "--feature_cols",
            help="List of feature columns to include in the dataset.",
        ),
    ] = ["stdorToU"],
    id_col: Annotated[
        str,
        typer.Option(
            "--id_col",
            help="Name of the household ID column.",
        ),
    ] = "LCLid",
    kwh_col: Annotated[
        str,
        typer.Option(
            "--kwh_col",
            help="Name of the kWh column.",
        ),
    ] = "KWH/hh (per half hour) ",
    datetime_col: Annotated[
        str,
        typer.Option(
            "--datetime_col",
            help="Name of the datetime column.",
        ),
    ] = "DateTime",
    utc: Annotated[
        bool,
        typer.Option(
            "--utc",
            help="Whether the datetime is in UTC.",
        ),
    ] = False,
    datetime_format: Annotated[
        str | None,
        typer.Option(
            "--datetime_format",
            help="Format of the datetime column, if not standard.",
        ),
    ] = None,
    historical_start: Annotated[
        str,
        typer.Option(
            "--historical_start",
            help="Start date for historical data (YYYY-MM-DD).",
        ),
    ] = "2012-01-01",
    historical_end: Annotated[
        str,
        typer.Option(
            "--historical_end",
            help="End date for historical data (YYYY-MM-DD).",
        ),
    ] = "2013-12-31",
    future_start: Annotated[
        str,
        typer.Option(
            "--future_start",
            help="Start date for future data (YYYY-MM-DD).",
        ),
    ] = "2014-01-01",
    future_end: Annotated[
        str,
        typer.Option(
            "--future_end",
            help="End date for future data (YYYY-MM-DD).",
        ),
    ] = "2014-12-31",
    drop_nulls: Annotated[
        bool,
        typer.Option(
            "--drop_nulls",
            help="Whether to drop rows with NaN kwh values. If False, will \
            replace NaN kwh values with 0.0",
        ),
    ] = True,
):
    """
    Split and preprocess your dataset.
    Default args are suitable for the LCL dataset. Modify as needed for other
    datasets.
    """

    get_data.split_preprocess_data(
        split,
        preprocess,
        data_dir,
        csv_data_path,
        sample_fraction,
        time_resolution,
        feature_cols,
        id_col,
        kwh_col,
        datetime_col,
        utc,
        datetime_format,
        historical_start,
        historical_end,
        future_start,
        future_end,
        drop_nulls,
    )


if __name__ == "__main__":
    app()
