from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

class ModelTrainer:
    def __init__(self, model_path='trained_model.joblib'):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.features = None
        self.target = None
        self.model_path = Path(model_path)

    def train(self, X, y):
        self.features = X
        self.target = y
        if len(X) < 10: #fix crossvalidation
            self.model = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=42)
        self.model.fit(self.features, self.target)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No saved model found at {self.model_path}")
            return False