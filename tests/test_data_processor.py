import unittest
import pandas as pd
import numpy as np
from TitanicClassifierCLI.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 1],
            'Pclass': [3, 1, 2],
            'Name': ['Juan Martinez', 'María García', 'Carla Rodriguez'],
            'Sex': ['male', 'female', 'female'],
            'Age': [22, 38, 26],
            'SibSp': [1, 1, 0],
            'Parch': [0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
            'Fare': [7.25, 71.2833, 7.925],
            'Cabin': ['', 'C85', ''],
            'Embarked': ['S', 'C', 'S']
        })
        self.data_processor = DataProcessor('dummy_path')
        self.data_processor.data = self.test_data

    def test_feature_engineering(self):
        self.data_processor.feature_engineering()
        
        # Check if new features were created
        self.assertIn('Title', self.data_processor.data.columns)
        self.assertIn('FamilySize', self.data_processor.data.columns)
        self.assertIn('IsAlone', self.data_processor.data.columns)

        # Check if specified columns have been dropped
        self.assertNotIn('Name', self.data_processor.data.columns)
        self.assertNotIn('Ticket', self.data_processor.data.columns)
        self.assertNotIn('Cabin', self.data_processor.data.columns)
        self.assertNotIn('PassengerId', self.data_processor.data.columns)
        self.assertNotIn('SibSp', self.data_processor.data.columns)
        self.assertNotIn('Parch', self.data_processor.data.columns)

    def test_preprocess(self):
        X, y = self.data_processor.preprocess()
        
        # Check if X is a numpy array
        self.assertIsInstance(X, np.ndarray)
        
        # Check if y is a pandas Series
        self.assertIsInstance(y, pd.Series)

        # Check if the number of features is correct
        expected_features = len(self.data_processor.get_feature_names())
        self.assertEqual(X.shape[1], expected_features)

if __name__ == '__main__':
    unittest.main()