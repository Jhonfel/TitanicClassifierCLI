from setuptools import setup, find_packages

setup(
    name="TitanicClassifierCLI",
    version="0.1",
    packages=find_packages(include=['TitanicClassifierCLI', 'TitanicClassifierCLI.*']),
    install_requires=[
        'Click',
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ], 
    entry_points='''
        [console_scripts]
        titanic-cli=TitanicClassifierCLI.cli:cli
    ''',
    author="Jhon Felipe Delgado Dalazar",
    author_email="jhonfeldel@gmail.com",
    description="A CLI tool for Titanic survival prediction",
    long_description="""
# Titanic Classifier CLI

## Description
Titanic Classifier CLI is a proof-of-concept machine learning project that predicts survival on the Titanic. It provides a command-line interface for training a model on the Titanic dataset and making predictions.

## Features
- Data preprocessing and cleaning
- Model training using RandomForest classifier
- Model evaluation with various metrics
- Prediction on new data
- Command-line interface for easy interaction
""",
    long_description_content_type="text/markdown",
    url="https://github.com/tuusuario/TitanicClassifierCLI",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)