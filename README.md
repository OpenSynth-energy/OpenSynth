# OpenSynth
OpenSynth Model Repository.

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
- pipenv run python app/app.py for a list of Typer app commands
- `get-lcl-data`: Downloads, Split, Preprocesses LCL dataset.

# 💽 Downloading Low Carbon London dataset [1] (https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)
- The compressed version of the data from data.london.gov.uk is ~ 700Mb. The full decompressed data is about 8Gb.
- Note: LCL data was compressed with compression algorithm that doesn't work with Python's `zipfile`. You'll need to manually unzip it via command line with `unzip` on Linux systems, or other equivalent on Windows machine.
- You can also download the low carbon london dataset using the typer app command `pipenv run python app/app.py --download`. This will use the subprocess module to unzip the file (for linux machines).
- If you're on windows, you'll need to manually download and unzip to the folder: `data/raw`
