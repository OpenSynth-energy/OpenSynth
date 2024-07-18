import logging
import os
import subprocess
from os import path

import wget


def check_redownload(file_path: str) -> bool:
    """
    Check if file already exists.
    If it does, ask user if they want to download it again.

    Args:
        file_path (str): file path

    Returns:
        bool: True if user expresses intent
        to download it again, False otherwise
    """
    if path.exists(file_path):
        prompt = f"{file_path} already exists. Redownload? (y/n): "
        return True if input(prompt) == "y" else False
    return True


def download_data(url: str, filename: str, unzip: bool = False):
    """
    Download data from a URL and save it to a file.
    Function will check if file already exists,
    and if does, prompt user to confirm redownload.

    Args:
        url (str): URL to download the data from
        filename (str): File path to save the downloaded data
    """
    if check_redownload(filename):
        if os.path.exists(filename):
            os.remove(filename)
        logging.info(f"Downloading data from: {url}")
        wget.download(url, filename)
    else:
        logging.info("Skipping download")

    if unzip:
        # Note: LCL dataset needs to be unzipped in command line with unzip
        # It's compressed using algorithms that Python's
        # zipfile module can't handle
        subprocess.run(["unzip", filename, "-d", "data/raw"])
