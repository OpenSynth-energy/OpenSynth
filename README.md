# [OpenSynth](https://lfenergy.org/projects/opensynth/)
OpenSynth Model Repository.


### For the data repository:
[_Link to CNZ's Synthetic Dataset on Zenodo here_](https://zenodo.org/records/13618456)

The data repository is still under construction. In the mean time, Centre for Net Zero has published a Faraday's output on Zenodo. This dataset contains 10 million synthetic load profiles of trained on over 300M smart meter readings from 20K Octopus Energy UK households sampled between 2021 and 2022, and is conditioned on labels such as the:
- Property types: house, flat, terraced, detached, semi-detached etc
- Energy performance certificate (EPC) rating: A/B/C, D/E, F/G etc
- Low Carbon Technology (LCT) ownership: heat pumps, electric vehicles, solar PVs etc
- Seasonality: days of the week and month of the year

You can find the dataset [here on Zenodo](https://zenodo.org/records/13618456). For more information about Faraday, please refer to the [workshop paper](https://arxiv.org/abs/2404.04314) that Centre for Net Zero presented at ICLR 2024. For more news and updates on OpenSynth, please subscribe to our mailing list [here](https://lists.lfenergy.org/g/opensynth-discussion).


# 💻 Development Set up

To set up environment for local development, you will need to set up PyEnv and Pipenv:
- [PyEnv](https://github.com/pyenv/pyenv) for Python versioning.
- [Pipenv](https://github.com/pypa/pipenv) for dependency management.

Then clone this repo and run `make setup`. This will set up all dependencies and precommit hooks.

Precommit Tools:
* [Pytest](https://github.com/pytest-dev/pytest/) for testing
* [Mypy](https://mypy.readthedocs.io/en/stable/) for type checking
* [Flake8](https://flake8.pycqa.org/en/latest/) for linting
* [isort](https://github.com/PyCQA/isort) for sorting imports
* [black](https://github.com/psf/black) for formatting


# Available CLI apps:
- `pipenv run python app/app.py` for a list of Typer app commands
- `get-lcl-data`: Downloads, Split, Preprocesses LCL dataset.


# 💽 Downloading Low Carbon London dataset [[1]](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
- The compressed version of the data from data.london.gov.uk is ~ 700Mb. The full decompressed data is about 8Gb.
- Note: LCL data was compressed with compression algorithm that doesn't work with Python's `zipfile`. You'll need to manually unzip it via command line with `unzip` on Linux systems, or other equivalent on Windows machine.
- You can also download the low carbon london dataset using the typer app command `pipenv run python app/app.py --download`. This will use the subprocess module to unzip the file (for linux machines).
- If you're on windows, you'll need to manually download and unzip to the folder: `data/raw`

### ℹ️ About Low Carbon London Dataset
- Low Carbon London dataset was from a trial conducted by UK Power Networks on a representative sample of London households from 2011 to 2014.
- The dataset contains half-hourly smart meter readings of 5,567 households.
- All timestamps are given in UTC so there's no time-zone conversation needed (i.e. 48 half-hourly data a day per household)

### ☁️ Preparing LCL Dataset for streaming
- In order to prepare the LCL Dataset for streaming, follow the instrucitons in `notebooks/streaming/streaming_data_preparation.ipynb`

# 📕 Tutorials
For tutorials on algorithms in this repository, please refer to notebooks in the `notebooks` folder.
- `faraday`: Train a synthetic data generative model using the Faraday algorithm
- `streaming`: Train a synthetic data generative model using the Faraday algorithm by streaming the training data (useful for out of memory datasets)
