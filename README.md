<p align="center">
  <img src="https://user-images.githubusercontent.com/1079118/124811147-623bd280-df1f-11eb-9f3a-a4a5e6ec5f94.png" alt="PyRIID">
</p>

[![Python](https://img.shields.io/pypi/pyversions/riid)](https://badge.fury.io/py/riid)
[![PyPI](https://badge.fury.io/py/riid.svg)](https://badge.fury.io/py/riid)

Welcome to PyRIID! PyRIID is a Python package providing models and data synthesis utilities supporting machine learning-based research into radioisotope-related detection, identification, and quantification.

## Installation

These instructions assume you meet the following requirements:

- Python version: 3.8 to 3.10
- Operating systems: Windows, Mac, or Ubuntu

A virtual environment is recommended.

Tests and examples are ran via Actions on many combinations of Python version and operating system.
You can verify support for your platform by checking the workflow files.

### For Use

To use the latest version on PyPI (note: changes are currently slower to appear here), run:

```sh
pip install riid
```

**For the latest features, run:**

```sh
pip install git+https://github.com/sandialabs/pyriid.git@main
```

### For Development

If you are developing PyRIID, clone this repository and run:

```sh
pip install -e ".[dev]"
```

**If you have trouble with Pylance resolving imports for an editable install, try this:**

```sh
pip install -e ".[dev]" --config-settings editable_mode=compat
```

## Examples

Examples for how to use this package can be found [here](https://github.com/sandialabs/PyRIID/blob/main/examples).

## Tests

Unit tests for this package can be found [here](https://github.com/sandialabs/PyRIID/blob/main/tests).

Run all unit tests with the following command:

```sh
python -m unittest tests/*.py -v
```

You can also run one of the `run_tests.*` scripts, whichever is appropriate for your platform.

## Docs

API documentation can be found [here](https://sandialabs.github.io/PyRIID).

You can also build the docs with the following commands:

```sh
pip install -r pdoc/requirements.txt
pdoc riid -o docs/ --html --template-dir pdoc
```

## Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and adhere to our [code of conduct](https://github.com/sandialabs/PyRIID/blob/main/CODE_OF_CONDUCT.md).

## Contacts

Maintainers and authors can be found [here](https://github.com/sandialabs/PyRIID/blob/main/pyproject.toml).

## Copyright

Full copyright details are outlined [here](https://github.com/sandialabs/PyRIID/blob/main/NOTICE.md)

## Acknowlegements

**Thank you** to the U.S. Department of Energy, National Nuclear Security Administration,
Office of Defense Nuclear Nonproliferation Research and Development (DNN R&D) for funding that has led to version `2.x`.

Additionally, **thank you** to the following individuals who have provided invaluable subject-matter expertise:

- Ben Maestas
- Greg Thoreson
- Michael Enghauser
- Elliott Leonard
