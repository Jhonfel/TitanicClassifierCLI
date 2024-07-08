import unittest
import numpy as np
from TitanicClassifierCLI.evaluator import Evaluator
from sklearn.ensemble import RandomForestClassifier

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.model = RandomForestClassifier(random_state=42)
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        self.y = np.array([0, 1, 1, 0])
        self.model.fit(self.X, self.y)
        self.evaluator = Evaluator(self.model)

    def test_evaluate(self):
        # Test if evaluation metrics are calculated correctly
        metrics = self.evaluator.evaluate(self.X, self.y)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertTrue(all(0 <= metric <= 1 for metric in metrics.values()))

    def test_plot_importances(self):
        # Test if feature importance plotting works without errors
        feature_names = ['Feature1', 'Feature2', 'Feature3']
        try:
            self.evaluator.plot_importances(feature_names)
        except Exception as e:
            self.fail(f"plot_importances raised {type(e).__name__} unexpectedly!")

    def test_plot_roc_curve(self):
        # Test if ROC curve plotting works without errors
        try:
            self.evaluator.plot_roc_curve(self.X, self.y)
        except Exception as e:
            self.fail(f"plot_roc_curve raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()