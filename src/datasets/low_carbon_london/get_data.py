import argparse
import logging
import sys

from src.datasets import datasets_utils
from src.datasets.low_carbon_london import preprocess_lcl, split_households

LCL_URL = "https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip"  # noqa
FILE_NAME = "data/raw/lcl_full_data.zip"  # noqa
CSV_FILE_NAME = "data/raw/CC_LCL-FullData.csv"

logging.basicConfig(level=logging.INFO)


def get_lcl_data(download: bool, split: bool, preprocess: bool):
    """
    Download, split and preprocess the Low Carbon London dataset.
    Download=True downloads and decompress data from data.london.gov.uk.
    Split=True splits the LCL households into training and holdout sets.
    Preprocess=True preprocesses the LCL data into daily load profiles.

    Notes:
    - The LCL dataset is a large dataset and may take a while to download.
    - The compressed version of LCL dataset is about 700MB, but the full
    dataset size is about 8GB.
    - Decompression may not work on windows. If so, please manually download
    and unzip the contents into the `data/raw` folder.

    Args:
        download (bool): True to download data from data.london.gov.uk.
        split (bool): True to split LCL households into training/ holdout sets.
        preprocess (bool): True to preprocess data.
    """
    if download:
        datasets_utils.download_data(LCL_URL, FILE_NAME)
    if split:
        split_households.split_lcl_data(CSV_FILE_NAME, 2000)
    if preprocess:
        preprocess_lcl.preprocess_lcl_data()


def parse_args(argument):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        required=False,
        help="Download and decompress the Low Carbon London"
        "dataset from data.london.gov.uk",
    )

    parser.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        default=False,
        required=False,
        help="Split the LCL households into training and holdout sets",
    )

    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        required=False,
        help="Preprocess the LCL data into daily load profiles",
    )

    return parser.parse_args(argument)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    DOWNLOAD = args.download
    SPLIT = args.split
    PREPROCESS = args.preprocess

    logging.info(
        f"Running get_lcl_data with download={DOWNLOAD}, "
        f"split={SPLIT}, preprocess={PREPROCESS}."
    )

    get_lcl_data(
        download=DOWNLOAD,
        split=SPLIT,
        preprocess=PREPROCESS,
    )
