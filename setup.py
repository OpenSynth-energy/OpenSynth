import os
import pathlib

from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

VERSION = "v0.0.5"


REPO_ROOT = pathlib.Path(__file__).parent

# Fetch the long description from the readme
with open(REPO_ROOT / "README.md", encoding="utf-8") as f:
    README = f.read()

install_requires = (
    "wget>=3.2",
    "tqdm>=4.66.4",
    "numpy<=2.0.0",
    "torch>=2.3.1",
    "scikit-learn>=1.3.2",
    "pytorch-lightning>=2.3.3",
    "matplotlib>3.6.1",
    "opacus>=1.4.1",
    "seaborn>=0.13.2",
    "torchmetrics>=1.4.1",
    "pandas>=1.2",
)

setup(
    name="opensynth-energy",
    version=VERSION,
    include_package_data=True,
    description="Opensynth is a library for"
    "synthetic energy demand generation.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Opensynth",
    author_email="pypi@opensynth.energy",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=install_requires,
    process_dependency_links=True,
    classifiers=[
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)
