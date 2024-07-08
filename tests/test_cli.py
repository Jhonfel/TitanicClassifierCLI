import unittest
from click.testing import CliRunner
from TitanicClassifierCLI.cli import cli, predict
import os
import pandas as pd
from unittest.mock import patch

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        
        # Create dummy CSV files for testing
        self.create_dummy_csv('test_train.csv')
        self.create_dummy_csv('test_test.csv')

    def tearDown(self):
        # Clean up dummy files after tests
        os.remove('test_train.csv')
        os.remove('test_test.csv')
        if os.path.exists('test_output.csv'):
            os.remove('test_output.csv')

    def create_dummy_csv(self, filename):
        data = {
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 1],
            'Pclass': [3, 1, 2],
            'Name': ['John Doe', 'Jane Doe', 'Alice'],
            'Sex': ['male', 'female', 'female'],
            'Age': [22, 38, 26],
            'SibSp': [1, 1, 0],
            'Parch': [0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
            'Fare': [7.25, 71.2833, 7.925],
            'Cabin': ['', 'C85', ''],
            'Embarked': ['S', 'C', 'S']
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

def test_cli_predict(self):
    result = self.runner.invoke(cli, ['predict', 
                                      '--train-data', 'test_train.csv',
                                      '--test-data', 'test_test.csv',
                                      '--model-path', 'test_model.joblib',
                                      '--output', 'test_output.csv',
                                      '--force-train'])
    self.assertEqual(result.exit_code, 0, f"Command failed with error: {result.output}")
    self.assertIn("Training a new model...", result.output)
    self.assertIn("Predictions completed and saved.", result.output)

@patch('TitanicClassifierCLI.cli.prompt_for_path')
def test_cli_predict_missing_file_with_prompt(self, mock_prompt):
    mock_prompt.side_effect = ['test_train.csv', 'test_test.csv', 'test_model.joblib', 'test_output.csv']
    result = self.runner.invoke(cli, ['predict', '--force-train'])
    self.assertEqual(result.exit_code, 0, f"Command failed with error: {result.output}")
    self.assertIn("Training a new model...", result.output)
    self.assertIn("Predictions completed and saved.", result.output)

@patch('TitanicClassifierCLI.cli.prompt_for_path')
def test_cli_predict_missing_file_with_invalid_prompt(self, mock_prompt):
    mock_prompt.side_effect = ['nonexistent.csv', 'test_test.csv', 'test_model.joblib', 'test_output.csv']
    result = self.runner.invoke(cli, ['predict'])
    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("File not found: nonexistent.csv", result.output)


if __name__ == '__main__':
    unittest.main()