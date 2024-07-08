import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.preprocessor = None
        self.fitted_preprocessor = None
        self.passenger_ids = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)
        if 'PassengerId' in self.data.columns:
            self.passenger_ids = self.data['PassengerId']
        return self.data

    def preprocess(self):
        self.data = self.data.copy()

        # Feature engineering
        self.feature_engineering()
    
        # Define column types
        numeric_features = ['Age', 'Fare', 'FamilySize']
        categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
    
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
    
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    
        # Fit and transform the data
        if 'Survived' in self.data.columns:
            X = self.data.drop('Survived', axis=1)
            y = self.data['Survived']
            X_transformed = self.preprocessor.fit_transform(X)
            self.fitted_preprocessor = self.preprocessor
            return X_transformed, y
        else:
            if self.fitted_preprocessor is None:
                raise ValueError("Preprocessor has not been fitted. Call preprocess on training data first.")
            X_transformed = self.fitted_preprocessor.transform(self.data)
            return X_transformed


    def feature_engineering(self):
        # Extract titles from names
        self.data['Title'] = self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_mapping = {"Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
                         "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Mlle": "Rare",
                         "Countess": "Rare", "Ms": "Rare", "Lady": "Rare", "Jonkheer": "Rare",
                         "Don": "Rare", "Mme": "Rare", "Capt": "Rare", "Sir": "Rare" }
        self.data['Title'] = self.data['Title'].map(title_mapping)

        # Create family size feature
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1

        # Create is_alone feature
        self.data['IsAlone'] = 0
        self.data.loc[self.data['FamilySize'] == 1, 'IsAlone'] = 1

        # Fill missing embarked with most common value
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])

        # Fill missing age with median age for title
        self.data['Age'] = self.data['Age'].fillna(self.data.groupby('Title')['Age'].transform('median'))
        # Fill missing fare with median fare for pclass
        self.data['Fare'] = self.data['Fare'].fillna(self.data.groupby('Pclass')['Fare'].transform('median'))
        # Log transform of fare
        self.data['Fare'] = np.log1p(self.data['Fare'])

        # Drop unnecessary columns
        self.data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)

    def get_feature_names(self):
        return self.preprocessor.get_feature_names_out()
    
    def get_fitted_preprocessor(self):
        return self.fitted_preprocessor
    
    def get_passenger_ids(self):
        return self.passenger_ids

