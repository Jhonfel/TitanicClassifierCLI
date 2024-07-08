from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.features = None
        self.target = None

    def train(self, X, y):
        self.features = X
        self.target = y
        self.model.fit(self.features, self.target)

    def predict(self, X):
        return self.model.predict(X)