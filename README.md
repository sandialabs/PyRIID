<p align="center">
  <img src="https://user-images.githubusercontent.com/1079118/124811147-623bd280-df1f-11eb-9f3a-a4a5e6ec5f94.png" alt="PyRIID">
</p>

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fsandialabs%2FPyRIID%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![PyPI](https://badge.fury.io/py/riid.svg)

PyRIID is a Python package providing modeling and data synthesis utilities for machine learning-based research and development of radioisotope-related detection, identification, and quantification.

## Installation

Requirements:

- Python version: 3.9 to 3.12
  - Note: we recommended the highest Python version you can manage as anecdotally, we have noticed that everything just tends to get faster.
- Operating systems: Windows, Mac, or Ubuntu

Tests and examples are run via Actions on many combinations of Python version and operating system.
You can verify support for your platform by checking the workflow files.

### For Use

To use the latest version on PyPI, run:

```sh
pip install riid
```

Note that changes are slower to appear on PyPI, so for the latest features, run:**

```sh
pip install git+https://github.com/sandialabs/pyriid.git@main
```

If you encounter Pylance issues, try:

```sh
pip install -e ".[dev]" --config-settings editable_mode=compat
```

### For Development

If you are developing PyRIID, clone this repository and run:

```sh
pip install -e ".[dev]"
```

## Examples

Examples for how to use this package can be found [here](https://github.com/sandialabs/PyRIID/blob/main/examples).

## Tests

Unit tests for this package can be found [here](https://github.com/sandialabs/PyRIID/blob/main/tests).

Run all unit tests with the following:

```sh
python -m unittest tests/*.py -v
```

You can also run one of the `run_tests.*` scripts, whichever is appropriate for your platform.

## Docs

API documentation can be found [here](https://sandialabs.github.io/PyRIID).

Docs can be built locally with the following:

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

Full copyright details can be found [here](https://github.com/sandialabs/PyRIID/blob/main/NOTICE.md).

## Acknowledgements

**Thank you** to the U.S. Department of Energy, National Nuclear Security Administration,
Office of Defense Nuclear Nonproliferation Research and Development (DNN R&D) for funding that has led to versions `2.0` and `2.1`.

Additionally, **thank you** to the following individuals who have provided invaluable subject-matter expertise:

- Paul Thelen (also an author)
- Ben Maestas
- Greg Thoreson
- Michael Enghauser
- Elliott Leonard

## Citing

When citing PyRIID, please reference the U.S. Department of Energy Office of Science and Technology Information (OSTI) record here:
[10.11578/dc.20221017.2](https://doi.org/10.11578/dc.20221017.2)

## Related Reports, Publications, and Projects

1. Alan Van Omen, *"A Semi-Supervised Model for Multi-Label Radioisotope Classification and Out-of-Distribution Detection."* Diss. 2023. doi: [10.7302/7200](https://dx.doi.org/10.7302/7200).
2. Tyler Morrow, *"Questionnaire for Radioisotope Identification and Estimation from Gamma Spectra using PyRIID v2."* United States: N. p., 2023. Web. doi: [10.2172/2229893](https://doi.org/10.2172/2229893).
3. Aaron Fjeldsted, Tyler Morrow, and Douglas Wolfe, *"Identifying Signal-to-Noise Ratios Representative of Gamma Detector Response in Realistic Scenarios,"* 2023 IEEE Nuclear Science Symposium, Medical Imaging Conference and International Symposium on Room-Temperature Semiconductor Detectors (NSS MIC RTSD), Vancouver, BC, Canada, 2023. doi: [10.1109/NSSMICRTSD49126.2023.10337860](https://doi.org/10.1109/NSSMICRTSD49126.2023.10337860).
4. Alan Van Omen and Tyler Morrow, *"A Semi-supervised Learning Method to Produce Explainable Radioisotope Proportion Estimates for NaI-based Synthetic and Measured Gamma Spectra."* United States: N. p., 2024. Web. doi: [10.2172/2335904](https://doi.org/10.2172/2335904).
    - [Code, data, and best model](https://zenodo.org/doi/10.5281/zenodo.10223445)
5. Alan Van Omen and Tyler Morrow, *"Controlling Radioisotope Proportions When Randomly Sampling from Dirichlet Distributions in PyRIID."* United States: N. p., 2024. Web. doi: [10.2172/2335905](https://doi.org/10.2172/2335905).
6. Alan Van Omen, Tyler Morrow, et al., *"Multilabel Proportion Prediction and Out-of-distribution Detection on Gamma Spectra of Short-lived Fission Products."* Annals of Nuclear Energy 208 (2024): 110777. doi: [10.1016/j.anucene.2024.110777](https://doi.org/10.1016/j.anucene.2024.110777).
    - [Code, data, and best models](https://zenodo.org/doi/10.5281/zenodo.12796964)
7. Aaron Fjeldsted, Tyler Morrow, et al., *"A Novel Methodology for Gamma-Ray Spectra Dataset Procurement over Varying Standoff Distances and Source Activities,"* Nuclear Instruments and Methods in Physics Research Section A (2024): 169681. doi: [10.1016/j.nima.2024.169681](https://doi.org/10.1016/j.nima.2024.169681).
