import unittest
import numpy as np
from TitanicClassifierCLI.model_trainer import ModelTrainer



import os

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.model_trainer = ModelTrainer('test_model.joblib')
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.y = np.array([0, 1, 1])

    def test_train(self):
        # Test if the model is trained successfully
        self.model_trainer.train(self.X, self.y)
        self.assertIsNotNone(self.model_trainer.model)

    def test_predict(self):
        # Test if predictions are made correctly
        self.model_trainer.train(self.X, self.y)
        predictions = self.model_trainer.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(all(isinstance(pred, (int, np.integer)) for pred in predictions))

    def test_save_load_model(self):
        # Test if the model can be saved and loaded correctly
        self.model_trainer.train(self.X, self.y)
        self.model_trainer.save_model()
        self.assertTrue(os.path.exists('test_model.joblib'))
        
        new_trainer = ModelTrainer('test_model.joblib')
        self.assertTrue(new_trainer.load_model())
        
        predictions = new_trainer.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))

    def tearDown(self):
        if os.path.exists('test_model.joblib'):
            os.remove('test_model.joblib')

if __name__ == '__main__':
    unittest.main()