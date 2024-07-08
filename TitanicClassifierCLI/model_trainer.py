from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.features = None
        self.target = None

    def train(self, X, y):
        self.features = X
        self.target = y
        self.model.fit(self.features, self.target)

    def predict(self, X):
        return self.model.predict(X)