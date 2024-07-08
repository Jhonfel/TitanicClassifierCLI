from setuptools import setup, find_packages

setup(
    name="TitanicClassifierCLI",
    version="0.1",
    packages=find_packages(include=['TitanicClassifierCLI', 'TitanicClassifierCLI.*']),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn"
    ], 
)