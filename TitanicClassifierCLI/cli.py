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

def setup_environment(train_data: Optional[str], test_data: Optional[str], output: str, model_path: str) -> None:
    """
    Set up the environment for the CLI.

    Args:
        train_data (str): Path to the training data.
        test_data (str): Path to the test data.
        output (str): Path to the output file.
        model_path (str): Path to the model file.
    """
    for path in [train_data, test_data]:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    output_dir = Path(output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the models directory exists
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)



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
@click.option('--model-path', default='trained_model.joblib', help='Path to save the model')
def train(train_data: str, model_path: str) -> None:
    """Train the model, evaluate it, and save it."""
    try:
        setup_environment(train_data, None, model_path)
        
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
        model_trainer = ModelTrainer(model_path)
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
        
        model_trainer.save_model()
        click.echo(f"Model trained and saved to {model_path}")

    except FileNotFoundError as e:
        click.echo(f"File not found: {str(e)}")
        raise click.Abort()
    except Exception as e:
        click.echo(f"An error occurred: {str(e)}")
        raise click.Abort()
    
@cli.command()
@click.option('--train-data', default='Data/train.csv', help='Path to training data')
@click.option('--test-data', default='Data/test.csv', help='Path to test data')
@click.option('--model-path', default='models/trained_model.joblib', help='Path to load/save the model')
@click.option('--output', default='submission.csv', help='Path to output predictions')
@click.option('--force-train', is_flag=True, help='Force training a new model')
def predict(train_data: str, test_data: str, model_path: str, output: str, force_train: bool) -> None:
    try:
        setup_environment(train_data, test_data, output, model_path)
        
        click.echo("Loading and preprocessing training data...")
        data_processor = DataProcessor(train_data)
        train_data = data_processor.load_data()
        X, y = data_processor.preprocess()
        
        model_trainer = ModelTrainer(model_path)
        if not model_trainer.load_model() or force_train:
            click.echo("Training a new model...")
            
            click.echo(f"Shape of training data after preprocessing: {X.shape}")
            click.echo(f"Feature names: {data_processor.get_feature_names()}")
            
            # Check class distribution
            click.echo("Class distribution in the full dataset:")
            click.echo(pd.Series(y).value_counts(normalize=True))
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            click.echo("Class distribution in the training set:")
            click.echo(pd.Series(y_train).value_counts(normalize=True))
            click.echo("Class distribution in the validation set:")
            click.echo(pd.Series(y_val).value_counts(normalize=True))
            
            model_trainer.train(X_train, y_train)
            
            click.echo("Evaluating model on validation set...")
            evaluator = Evaluator(model_trainer.model)
            evaluator.evaluate(X_val, y_val)
            
            click.echo("Confusion matrix:")
            evaluator.print_confusion_matrix(y_val, model_trainer.predict(X_val))
            
            if len(X) >= 5:  # Only perform cross-validation if there are enough samples
                cv_scores = cross_val_score(model_trainer.model, X, y, cv=min(5, len(X)))
                click.echo(f"Cross-validation scores: {cv_scores}")
                click.echo(f"Mean cross-validation score: {cv_scores.mean()}")
            
            model_trainer.save_model()
            click.echo(f"Model trained and saved to {model_path}")
        else:
            click.echo(f"Model loaded from {model_path}")
        
        click.echo("Processing test data...")
        test_processor = DataProcessor(test_data)
        test_data = test_processor.load_data()
        test_processor.fitted_preprocessor = data_processor.get_fitted_preprocessor()
        X_test = test_processor.preprocess()
        
        click.echo("Making predictions...")
        predictions = model_trainer.predict(X_test)
        
        click.echo(f"Saving predictions to {output}")
        passenger_ids = test_processor.get_passenger_ids()
        submission = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
        submission.to_csv(output, index=False)
        click.echo("Predictions completed and saved.")

    except FileNotFoundError as e:
        click.echo(f"File not found: {str(e)}")
        raise click.Abort()
    except Exception as e:
        click.echo(f"An error occurred: {str(e)}")
        raise click.Abort()

if __name__ == '__main__':
    cli()