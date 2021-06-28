"""Simple setup script for installing the core package.
"""
from setuptools import setup, find_packages

setup(
    name="riid",
    version="1.0.0",
    description="Machine learning-based models and utilities for radioisotope identification",
    author="Tyler Morrow,Nathan Price",
    author_email="tmorro@sandia.gov,njprice@sandia.gov",
    url="https://www.sandia.gov",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.0.0",
        "tensorflow-model-optimization==0.1.3",
        "numpy==1.17.4",
        "pandas==1.0.5",
        "matplotlib==3.1.2",
        "scikit-learn==0.22",
        "tables==3.6.1",
        "tqdm",
        "seaborn==0.10.1"
    ],
)
