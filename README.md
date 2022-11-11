<p align="center">
  <img src="https://user-images.githubusercontent.com/1079118/124811147-623bd280-df1f-11eb-9f3a-a4a5e6ec5f94.png" alt="PyRIID">
</p>

[![Python](https://img.shields.io/pypi/pyversions/riid)](https://badge.fury.io/py/riid)
[![PyPI](https://badge.fury.io/py/riid.svg)](https://badge.fury.io/py/riid)

This repository contains the PyRIID package (as well as tests and examples) which is intended to provide utilities that support machine learning-based research and solutions to radioisotope identification.

## Installation

To use the latest version in your own project, run:

```
pip install riid
```

For development, clone this repository and run:

```
pip install -e ".[dev]"
```

## Examples

Check out the `./examples` folder for numerous examples on how to use this package.

## Tests

Run all unit tests with the following command:

```sh
python -m unittest tests/*.py -v
```

You can also run one of the `run_tests.*` scripts, whichever is appropriate for your platform.

## Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and adhere to our [code of conduct](https://github.com/sandialabs/PyRIID/blob/main/CODE_OF_CONDUCT.md).

## Contacts

Maintainers and authors can be found [here](https://github.com/sandialabs/PyRIID/blob/main/pyproject.toml).

## Copyright

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

This source code is licensed under the BSD-style license found [here](https://github.com/sandialabs/PyRIID/blob/main/LICENSE.md).

## Acknowlegements

**Thank you** to the U.S. Department of Energy, National Nuclear Security Administration,
Office of Defense Nuclear Nonproliferation Research and Development (DNN R&D) for funding that has led to version `2.x`.

Additionally, **thank you** to the following individuals who have provided invaluable subject-matter expertise:

- Ben Maestas
- Greg Thoreson
- Michael Enghauser
