# 🚢 Titanic Classifier CLI

## 📜 Description
Titanic Classifier CLI is a proof of concept machine learning project that predicts survival on the Titanic. It provides a command-line interface for training a model on the Titanic dataset and making predictions. This project serves as a demonstration of MLOps practices and CLI tool development in Python.

## 📋 Table of Contents
- [🚢 Titanic Classifier CLI](#-titanic-classifier-cli)
  - [📜 Description](#-description)
  - [📋 Table of Contents](#-table-of-contents)
  - [🛠️ Installation](#️-installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
  - [🚀 Usage](#-usage)
  - [📁 Project Structure](#-project-structure)
  - [🧪 Testing](#-testing)
  - [🚀 Deployment](#-deployment)

## 🛠️ Installation

### Prerequisites
- Python 3.7+ 🐍
- pip 📦

### Steps
1. Clone the repository:
```
git clone https://github.com/yourusername/TitanicClassifierCLI.git
https://github.com/Jhonfel/TitanicClassifierCLI.git
```

2. Install the package:
```
pip install -e .
```
3. Alternatively, you can install the package directly from PyPI:
```
pip install TitanicClassifierCLI
```


## 🚀 Usage

To use the Titanic Classifier CLI:
```
titanic-cli predict --train-data path/to/train.csv --test-data path/to/test.csv --output predictions.csv
```

Options:
- `--train-data`: Path to the training data CSV file (default: 'Data/train.csv')
- `--test-data`: Path to the test data CSV file (default: 'Data/test.csv')
- `--output`: Path to save the output predictions CSV file (default: 'submission.csv')

## 📁 Project Structure
```
TitanicClassifierCLI/
│
├── Data/
│   ├── train.csv
│   └── test.csv
├── TitanicClassifierCLI/
│   ├── init.py
│   ├── cli.py
│   ├── data_processor.py
│   ├── evaluator.py
│   └── model_trainer.py
├── tests/
│   ├── test_cli.py
│   ├── test_data_processor.py
│   ├── test_evaluator.py
│   └── test_model_trainer.py
├── Notebooks/
│   └── data_cleaning_and_ml_model_evaluation.ipynb
├── README.md
├── setup.py
└── requirements.txt
```

## 🧪 Testing

To run tests and generate a coverage report:
coverage run -m unittest discover tests
coverage report -m
coverage html  # generates a detailed HTML report

###current coverage
```
Name                                     Stmts   Miss  Cover   Missing
----------------------------------------------------------------------
TitanicClassifierCLI/__init__.py             5      0   100%
TitanicClassifierCLI/cli.py                149     23    85%   48-54, 108, 123-130, 205-207, 209-211, 218
TitanicClassifierCLI/data_processor.py      74      7    91%   43, 75, 80-84
TitanicClassifierCLI/evaluator.py           61      9    85%   43-49, 62, 82
TitanicClassifierCLI/model_trainer.py       27      0   100%
tests/test_cli.py                          108      6    94%   161-162, 203-205, 232
tests/test_data_processor.py                28      1    96%   55
tests/test_evaluator.py                     31      5    84%   28-29, 35-36, 39
tests/test_model_trainer.py                 30      1    97%   44
----------------------------------------------------------------------
TOTAL                                      513     52    90%
```

## 🚀 Deployment

This project uses Docker for containerization and can be deployed using the following steps:

1. Build the Docker image:
```
docker build -t titanic-classifier-cli .
```

2. Run the Docker container:
```
docker run -v /path/to/your/data:/app/data titanic-classifier-cli predict --train-data /app/data/train.csv --test-data /app/data/test.csv --output /app/data/predictions.csv
```

---

⚠️ **Note**: This is a proof of concept project and is not intended for production use without further development and testing.
