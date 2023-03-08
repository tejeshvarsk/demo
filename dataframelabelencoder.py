
from sklearn.base import TransformerMixin
from collections import defaultdict
from  category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

class DataFrameLabelEncoder(TransformerMixin):
    def __init__(self):
        self.label_encoders = defaultdict(LabelEncoder)
    
    def fit(self, X):
        for column in X.columns:
            if X[column].dtypes.name in ('category', 'object'):
                self.label_encoders[column] = OrdinalEncoder()
                self.label_encoders[column].fit(X[column])
        return self
    
    def transform(self, X):
        for column, label_encoder in self.label_encoders.items():
            X[column] = label_encoder.transform(X[column])
        return X
