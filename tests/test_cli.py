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
                                          '--output', 'test_output.csv'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Loading and preprocessing training data...", result.output)
        self.assertIn("Training model...", result.output)
        self.assertIn("Evaluating model on validation set...", result.output)
        self.assertIn("Processing test data and making predictions...", result.output)
        self.assertIn("Saving predictions to test_output.csv", result.output)
        self.assertTrue(os.path.exists('test_output.csv'))

    @patch('TitanicClassifierCLI.cli.prompt_for_path')
    def test_cli_predict_missing_file_with_prompt(self, mock_prompt):
        mock_prompt.side_effect = ['test_train.csv', 'test_test.csv', 'test_output.csv']
        result = self.runner.invoke(cli, ['predict'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Loading and preprocessing training data...", result.output)
        self.assertIn("Training model...", result.output)
        self.assertIn("Evaluating model on validation set...", result.output)
        self.assertIn("Processing test data and making predictions...", result.output)
        self.assertIn("Saving predictions to test_output.csv", result.output)
        self.assertTrue(os.path.exists('test_output.csv'))
        self.assertEqual(mock_prompt.call_count, 3)

    def test_cli_predict_invalid_option(self):
        result = self.runner.invoke(cli, ['predict', '--invalid-option'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error: No such option: --invalid-option", result.output)

    @patch('TitanicClassifierCLI.cli.prompt_for_path')
    def test_cli_predict_missing_file_with_invalid_prompt(self, mock_prompt):
        mock_prompt.side_effect = ['nonexistent.csv', 'test_test.csv', 'test_output.csv']
        result = self.runner.invoke(cli, ['predict'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("File not found: nonexistent.csv", result.output)

if __name__ == '__main__':
    unittest.main()