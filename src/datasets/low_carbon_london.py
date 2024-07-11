import logging

from src.datasets import datasets_utils

logging.basicConfig(level=logging.INFO)

LCL_URL = "https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip"  # noqa
FILE_NAME = "data/raw/lcl_full_data.zip"  # noqa


def get_lcl_data(download: bool, split: bool, preprocess: bool):
    if download:
        datasets_utils.download_data(LCL_URL, FILE_NAME)
    if split:
        pass
    if preprocess:
        pass


if __name__ == "__main__":
    logging.info(f"Downloading data from {LCL_URL}")
    get_lcl_data(download=True, split=False, preprocess=False)
