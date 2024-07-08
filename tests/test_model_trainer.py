import unittest
import numpy as np
from TitanicClassifierCLI.model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.model_trainer = ModelTrainer()
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

if __name__ == '__main__':
    unittest.main()