![PyRIID](https://repository-images.githubusercontent.com/381080801/22d4fe00-d81c-11eb-92c4-a71fc2e348a3)

This repository contains core functions and classes used by the BALDR project (Base Algorithms for Learned Detection of Radioisotopes) for its research in machine learning-based radioisotope identification (ML-RIID).

## Prerequisites

- Python 3.7+

## Installation

This section will assume you have:

- installed Python 3.7 or higher; and
- cloned this repo

With the above done, perform the following:

```
pip install <path to repository>
```

### Data Directory (optional)

Some functions are usable only if you set the `PYRIID_DATA_DIR` environment variable to a path to some directory on your computer.

## Examples

Check out the `./examples` folder for numerous examples on how to use this package.

## Tests

Run all unit tests with the following command:

```sh
python -m unittest tests/*.py -v
```

Or you run the `run_tests.sh` script.

## Contributing

Pull requests are welcome.
For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate and adhere to our [code of conduct](./CODE_OF_CONDUCT.md).

## Authors

* Tyler Morrow - tmorro@sandia.gov
* Nathan Price - njprice@sandia.gov

## Copyright

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

This source code is licensed under the BSD-style license found [here](./LICENSE.md).
