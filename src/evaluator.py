from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        accuracy = accuracy_score(y, predictions)
        print(f"Accuracy: {accuracy:.2f}")

    def plot_importances(self, feature_names):
        importances = self.model.feature_importances_
        indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)
        plt.barh([feature_names[i] for i in indices], [importances[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.show()