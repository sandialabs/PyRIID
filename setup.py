# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
"""Simple setup script for installing the core package.
"""
from os import path
from setuptools import find_packages, setup


p = path.abspath(path.dirname(__file__))
with open(path.join(p, './README.md')) as f:
    README = f.read()

setup(
    name="riid",
    version="1.0.2",
    description="Machine learning-based models and utilities for radioisotope identification",
    long_description=README,
    long_description_content_type='text/markdown',
    author="Tyler Morrow,Nathan Price",
    author_email="tmorro@sandia.gov,njprice@sandia.gov",
    url="https://github.com/sandialabs/PyRIID",
    packages=find_packages(),
    python_requires=">=3.7, <3.8",
    install_requires=[
        "tensorflow==2.0.0",
        "tensorflow-model-optimization==0.1.3",
        "numpy==1.17.4",
        "pandas==1.0.5",
        "matplotlib==3.1.2",
        "scikit-learn==0.22",
        "tables==3.6.1",
        "tqdm",
        "seaborn==0.10.1",
        "h5py<3.0.0",
    ],
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7'
    ],
    license='BSD-3',
    keywords='pyriid riid machine learning',
)
