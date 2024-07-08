import unittest
from click.testing import CliRunner
from TitanicClassifierCLI.cli import cli, predict, train, setup_environment
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import joblib
from sklearn.dummy import DummyClassifier
from pathlib import Path
import sys
import click
import logging


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        
        # Create dummy CSV files for testing
        self.create_dummy_csv('test_train.csv')
        self.create_dummy_csv('test_test.csv')
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def tearDown(self):
        # Clean up dummy files after tests
        for file in ['test_train.csv', 'test_test.csv', 'test_output.csv', 'test_model.joblib']:
            if os.path.exists(file):
                os.remove(file)

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
        # Crear un conjunto de datos más grande
        np.random.seed(42)
        n_samples = 100
        train_data = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.randint(0, 2, n_samples),
            'Pclass': np.random.randint(1, 4, n_samples),
            'Name': [f'Passenger {i}' for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.randint(1, 80, n_samples),
            'SibSp': np.random.randint(0, 5, n_samples),
            'Parch': np.random.randint(0, 5, n_samples),
            'Ticket': [f'Ticket {i}' for i in range(1, n_samples + 1)],
            'Fare': np.random.uniform(10, 100, n_samples),
            'Cabin': [''] * n_samples,
            'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples)
        })
        train_data.to_csv('test_train.csv', index=False)
    
        test_data = train_data.drop('Survived', axis=1).head(20)  # Usar solo 20 muestras para el test
        test_data.to_csv('test_test.csv', index=False)
    
        result = self.runner.invoke(cli, ['predict', 
                                      '--train-data', 'test_train.csv',
                                      '--test-data', 'test_test.csv',
                                      '--model-path', 'test_model.joblib',
                                      '--output', 'test_output.csv',
                                      '--force-train'])
    
        print(f"Command output: {result.output}")  # Para depuración
        self.assertEqual(result.exit_code, 0, f"Command failed with error: {result.output}")
        
        # Verificar que se creó el archivo de salida
        self.assertTrue(Path('test_output.csv').exists())
        # Limpiar archivos creados
        for file in ['test_train.csv', 'test_test.csv', 'test_model.joblib', 'test_output.csv']:
            Path(file).unlink(missing_ok=True)

    def test_cli_train(self):
        # Crear un conjunto de datos de entrenamiento con ambas clases
        np.random.seed(42)
        n_samples = 100
        train_data = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.randint(0, 2, n_samples),  # Asegura que haya tanto 0 como 1
            'Pclass': np.random.randint(1, 4, n_samples),
            'Name': [f'Passenger {i}' for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.randint(1, 80, n_samples),
            'SibSp': np.random.randint(0, 5, n_samples),
            'Parch': np.random.randint(0, 5, n_samples),
            'Ticket': [f'Ticket {i}' for i in range(1, n_samples + 1)],
            'Fare': np.random.uniform(10, 100, n_samples),
            'Cabin': [''] * n_samples,
            'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples)
        })
        
        # Asegurarse de que haya al menos una muestra de cada clase
        train_data.loc[0, 'Survived'] = 0
        train_data.loc[1, 'Survived'] = 1
        
        train_data.to_csv('test_train.csv', index=False)
    
        result = self.runner.invoke(cli, ['train', 
                                          '--train-data', 'test_train.csv',
                                          '--model-path', 'test_model.joblib'])
        
        print(f"Command output: {result.output}")  # Para depuración
        self.assertEqual(result.exit_code, 0, f"Command failed with error: {result.output}")
        self.assertIn("Model saved to", result.output)
        self.assertTrue(Path('test_model.joblib').exists())
    
        # Limpiar archivos creados
        for file in ['test_train.csv', 'test_model.joblib']:
            Path(file).unlink(missing_ok=True)

    def test_cli_train_file_not_found(self):
        result = self.runner.invoke(cli, ['train', 
                                          '--train-data', 'nonexistent.csv',
                                          '--model-path', 'test_model.joblib'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("File not found", result.output)

    @patch('TitanicClassifierCLI.cli.DataProcessor')
    @patch('TitanicClassifierCLI.cli.ModelTrainer')
    def test_cli_predict_data_processing(self, mock_model_trainer, mock_data_processor):
        # Configura el mock de DataProcessor
        mock_data_processor.return_value.load_data.return_value = pd.DataFrame({'PassengerId': [1, 2, 3]})
        mock_data_processor.return_value.preprocess.side_effect = [
            (pd.DataFrame({'feature': [1, 2, 3]}), pd.Series([0, 1, 0])),  # Para los datos de entrenamiento
            pd.DataFrame({'feature': [1, 2, 3]})  # Para los datos de prueba
        ]
        mock_data_processor.return_value.get_passenger_ids.return_value = [1, 2, 3]
        
        # Configura el mock de ModelTrainer
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 1, 1]
        mock_model_trainer.return_value.model = mock_model
        mock_model_trainer.return_value.predict.return_value = [1, 1, 1]
        mock_model_trainer.return_value.load_model.return_value = True
    
        result = self.runner.invoke(cli, ['predict', 
                                          '--train-data', 'test_train.csv',
                                          '--test-data', 'test_test.csv',
                                          '--model-path', 'test_model.joblib',
                                          '--output', 'test_output.csv'])
        
        print(f"Command output: {result.output}")
        
        if result.exception:
            import traceback
            print(traceback.format_exc())
    
        self.assertEqual(result.exit_code, 0, f"Command failed with error: {result.output}")
        mock_data_processor.return_value.load_data.assert_called()
        mock_data_processor.return_value.preprocess.assert_called()
        mock_model_trainer.return_value.predict.assert_called_once()
        mock_data_processor.return_value.get_passenger_ids.assert_called_once()
        self.assertTrue(Path('test_output.csv').exists())

    def test_cli_predict_without_force_train(self):
        # Crear un modelo dummy y guardarlo
        dummy_model = DummyClassifier(strategy="constant", constant=1)
        dummy_model.fit(pd.DataFrame({'feature': [1, 2, 3]}), [1, 1, 1])
        joblib.dump(dummy_model, 'test_model.joblib')
    
        # Crear archivos CSV de prueba
        train_data = pd.DataFrame({
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
        })
        train_data.to_csv('test_train.csv', index=False)
    
        test_data = train_data.drop('Survived', axis=1)
        test_data.to_csv('test_test.csv', index=False)
    
        result = self.runner.invoke(cli, ['predict', '--train-data', 'test_train.csv', '--test-data', 'test_test.csv', '--model-path', 'test_model.joblib', '--output', 'test_output.csv'], catch_exceptions=False)

        
        print(f"Command output: {result.output}")
        print(f"Exit code: {result.exit_code}")
        if result.exception:
            print(f"Exception: {result.exception}")
            import traceback
            print(f"Traceback: {''.join(traceback.format_tb(result.exc_info[2]))}")

        
        self.assertEqual(result.exit_code, 0, f"Command failed with error: {result.output}")
        self.assertIn("Model loaded from", result.output)

        # Verificar que se creó el archivo de salida
        self.assertTrue(Path('test_output.csv').exists())
    
        # Limpiar archivos creados
        for file in ['test_train.csv', 'test_test.csv', 'test_model.joblib', 'test_output.csv']:
            Path(file).unlink(missing_ok=True)

    @patch('TitanicClassifierCLI.cli.DataProcessor')
    def test_cli_predict_exception_handling(self, mock_data_processor):
        mock_data_processor.side_effect = Exception("Test exception")
        result = self.runner.invoke(cli, ['predict', 
                                          '--train-data', 'test_train.csv',
                                          '--test-data', 'test_test.csv',
                                          '--model-path', 'test_model.joblib',
                                          '--output', 'test_output.csv'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("An unexpected error occurred: Test exception", result.output)



if __name__ == '__main__':
    unittest.main()