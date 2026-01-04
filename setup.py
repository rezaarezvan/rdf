#!/usr/bin/env python3
from pathlib import Path
from setuptools import setup, find_packages

directory = Path(__file__).resolve().parent
with open(directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rdf",
    version="0.2.0",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={"rdf": ["academic.mplstyle"]},
    entry_points={"console_scripts": ["rdf = rdf.__main__:main"]},
    classifiers=["Programming Language :: Python :: 3"],
    python_requires=">=3.10",
    install_requires=["matplotlib>=3.5"],
    include_package_data=True,
)
