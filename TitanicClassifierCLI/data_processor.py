import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.categorical_cols = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        return self.data

    def preprocess(self):
        self.data.drop(columns=['Cabin', 'Fare', 'Ticket', 'Name'], inplace=True)
        self.encode_categorical()
        self.handle_missing_values()

    def handle_missing_values(self):
        imputer = SimpleImputer(strategy='mean')
        self.data.iloc[:, :] = imputer.fit_transform(self.data)

    def encode_categorical(self):
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns
        ordinal_encoder = OrdinalEncoder()
        self.data[self.categorical_cols] = ordinal_encoder.fit_transform(self.data[self.categorical_cols])
