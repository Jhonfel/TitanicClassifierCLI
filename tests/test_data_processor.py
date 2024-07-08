import unittest
import pandas as pd
from TitanicClassifierCLI.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create a test DataFrame
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

    def test_preprocess(self):
        self.data_processor.preprocess()
        
        # Check if specified columns have been dropped
        self.assertNotIn('Cabin', self.data_processor.data.columns)
        self.assertNotIn('Fare', self.data_processor.data.columns)
        self.assertNotIn('Ticket', self.data_processor.data.columns)
        self.assertNotIn('Name', self.data_processor.data.columns)

        # Check if categorical columns have been encoded
        self.assertTrue(self.data_processor.data['Sex'].dtype != 'object')
        self.assertTrue(self.data_processor.data['Embarked'].dtype != 'object')

        # Check if there are no null values
        self.assertTrue(self.data_processor.data.isnull().sum().sum() == 0)

if __name__ == '__main__':
    unittest.main()