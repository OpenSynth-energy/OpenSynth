from src.datasets import datasets_utils
from src.datasets.low_carbon_london import split_households

LCL_URL = "https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip"  # noqa
FILE_NAME = "data/raw/lcl_full_data.zip"  # noqa
CSV_FILE_NAME = "data/raw/CC_LCL-FullData.csv"


def get_lcl_data(download: bool, split: bool, preprocess: bool):
    if download:
        datasets_utils.download_data(LCL_URL, FILE_NAME)
    if split:
        split_households.split_lcl_data(CSV_FILE_NAME, 2000)
    if preprocess:
        pass


if __name__ == "__main__":
    get_lcl_data(
        download=False,
        split=True,
        preprocess=False,
    )
