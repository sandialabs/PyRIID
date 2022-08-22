# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""Simple setup script for installing the core package."""
from os import path
from sys import platform

from setuptools import find_packages, setup

p = path.abspath(path.dirname(__file__))
with open(path.join(p, './README.md')) as f:
    README = f.read()

REQUIREMENTS = [
    "tensorflow==2.5.0",
    "tensorflow-probability==0.12.1",
    "tensorflow-model-optimization==0.6.0",
    "numpy~=1.19.2",
    "pandas==1.0.5",
    "matplotlib==3.5.2",
    "scikit-learn==0.24",
    "tables==3.6.1",
    "seaborn==0.11.1",
]
if platform == "win32":
    REQUIREMENTS.append("pythonnet==2.5.2")
    REQUIREMENTS.append("PyYAML==6.0")

setup(
    name="riid",
    description="Machine learning-based models and utilities for radioisotope identification",
    long_description=README,
    long_description_content_type='text/markdown',
    author="Tyler Morrow,Aislinn Handley,Alan Van Omen",
    author_email="tmorro@sandia.gov,ajhandl@sandia.gov,ajvanom@sandia.gov",
    url="https://github.com/sandialabs/PyRIID",
    packages=find_packages(),
    python_requires=">=3.7, <3.8",
    install_requires=REQUIREMENTS,
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
