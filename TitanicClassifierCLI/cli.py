import click
import logging
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import pandas as pd
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
    """Train model and make predictions."""
    try:
        setup_environment(train_data, test_data, output)
        
        click.echo("Loading and preprocessing training data...")
        data_processor = DataProcessor(train_data)
        train_data = data_processor.load_data()
        X, y = data_processor.preprocess()
        
        click.echo(f"Shape of training data after preprocessing: {X.shape}")
        click.echo(f"Feature names: {data_processor.get_feature_names()}")
        
        # Check class distribution
        click.echo("Class distribution in the full dataset:")
        click.echo(pd.Series(y).value_counts(normalize=True))
        
        click.echo("Splitting data into train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        click.echo("Class distribution in the training set:")
        click.echo(pd.Series(y_train).value_counts(normalize=True))
        click.echo("Class distribution in the validation set:")
        click.echo(pd.Series(y_val).value_counts(normalize=True))
        
        click.echo("Training model...")
        model_trainer = ModelTrainer()
        model_trainer.train(X_train, y_train)
        
        click.echo("Evaluating model on validation set...")
        evaluator = Evaluator(model_trainer.model)
        evaluator.evaluate(X_val, y_val)
        
        # Plot confusion matrix
        click.echo("Confusion matrix:")
        evaluator.print_confusion_matrix(y_val, model_trainer.predict(X_val))
        
        # Perform cross-validation
        cv_scores = cross_val_score(model_trainer.model, X, y, cv=5)
        click.echo(f"Cross-validation scores: {cv_scores}")
        click.echo(f"Mean cross-validation score: {cv_scores.mean()}")
        
        # Compare with Logistic Regression
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train, y_train)
        click.echo("Metrics for Logistic Regression:")
        evaluator_log = Evaluator(log_reg)
        evaluator_log.evaluate(X_val, y_val)
        
        click.echo("Processing test data and making predictions...")
        test_processor = DataProcessor(test_data)
        test_data = test_processor.load_data()
        test_processor.fitted_preprocessor = data_processor.get_fitted_preprocessor()
        X_test = test_processor.preprocess()
        
        click.echo(f"Shape of test data after preprocessing: {X_test.shape}")
        
        if X_test.shape[1] != X_train.shape[1]:
            click.echo(f"Warning: Number of features in test data ({X_test.shape[1]}) "
                       f"does not match training data ({X_train.shape[1]})")
        
        predictions = model_trainer.predict(X_test)
        
        click.echo(f"Saving predictions to {output}")
        passenger_ids = test_processor.get_passenger_ids()
        if passenger_ids is None:
            click.echo("Warning: PassengerId not found in test data. Using index as PassengerId.")
            passenger_ids = range(1, len(predictions) + 1)
        
        submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
        submission.to_csv(output, index=False)
        click.echo("Done!")

    except Exception as e:
        click.echo(f"An error occurred: {str(e)}")
        raise


if __name__ == '__main__':
    cli()