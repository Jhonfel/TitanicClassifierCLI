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
)