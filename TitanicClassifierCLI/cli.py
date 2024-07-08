import click
import logging
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split

from .data_processor import DataProcessor
from .model_trainer import ModelTrainer
from .evaluator import Evaluator

# Config logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment(train_data: str, test_data: str, output: str) -> None:
    """
    Set up the environment for the CLI.

    Args:
        train_data (str): Path to the training data.
        test_data (str): Path to the test data.
        output (str): Path to the output file.
    """
    for path in [train_data, test_data]:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    output_dir = Path(output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

def prompt_for_path(path_type: str) -> str:
    """
    Prompt the user for a file path.

    Args:
        path_type (str): The type of path (e.g., 'train data', 'test data', 'output').

    Returns:
        str: The user-provided file path.
    """
    while True:
        user_input = click.prompt(f"Enter the path for the {path_type}", type=str)
        if Path(user_input).exists() or path_type == 'output':
            return user_input
        click.echo(f"The specified path does not exist. Please try again.")

@click.group()
def cli():
    """Titanic Classifier CLI"""
    pass

@cli.command()
@click.option('--train-data', default='Data/train.csv', help='Path to training data')
@click.option('--test-data', default='Data/test.csv', help='Path to test data')
@click.option('--output', default='submission.csv', help='Path to output predictions')
def predict(train_data: str, test_data: str, output: str) -> None:
    """
    Train model and make predictions.

    Args:
        train_data (str): Path to the training data.
        test_data (str): Path to the test data.
        output (str): Path to save the predictions.
    """
    try:
        # Check if default paths exist, if not, prompt for input
        if not Path(train_data).exists():
            train_data = prompt_for_path('train data')
        if not Path(test_data).exists():
            test_data = prompt_for_path('test data')
        if not Path(output).parent.exists():
            output = prompt_for_path('output')

        setup_environment(train_data, test_data, output)

        logger.info("Loading and preprocessing training data...")
        data_processor = DataProcessor(train_data)
        data = data_processor.load_data()
        data_processor.preprocess()

        logger.info("Splitting data into train and validation sets...")
        X = data.drop('Survived', axis=1)
        y = data['Survived']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Training model...")
        model_trainer = ModelTrainer()
        model_trainer.train(X_train, y_train)

        logger.info("Evaluating model on validation set...")
        evaluator = Evaluator(model_trainer.model)
        evaluator.evaluate(X_val, y_val)

        logger.info("Processing test data and making predictions...")
        test_processor = DataProcessor(test_data)
        test_data = test_processor.load_data()
        test_processor.preprocess()
        predictions = model_trainer.predict(test_data)

        logger.info(f"Saving predictions to {output}")
        test_data['Survived'] = predictions
        test_data[['PassengerId', 'Survived']].to_csv(output, index=False)
        logger.info("Done!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    cli()