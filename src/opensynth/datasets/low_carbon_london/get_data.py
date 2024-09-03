# Copyright Contributors to the Opensynth-energy Project.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from opensynth.datasets import datasets_utils
from opensynth.datasets.low_carbon_london import (
    preprocess_lcl,
    split_households,
)

LCL_URL = "https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip"  # noqa
FILE_NAME = Path("data/raw/lcl_full_data.zip")  # noqa
CSV_FILE_NAME = Path("data/raw/CC_LCL-FullData.csv")  # noqa

logger = logging.getLogger(__name__)


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
    logger.info(
        f"Running get_lcl_data with download={download}, "
        f"split={split}, preprocess={preprocess}."
    )

    if download:
        datasets_utils.download_data(LCL_URL, FILE_NAME)
    if split:
        split_households.split_lcl_data(CSV_FILE_NAME, 2000)
    if preprocess:
        preprocess_lcl.preprocess_lcl_data()
