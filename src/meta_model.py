import numpy as np
from sklearn.linear_model import LinearRegression

class MetaModel:
    
    def __init__(self):
        self.model = LinearRegression()
        self.min_value = None
        self.max_value = None
    
    def fit(self, df, target="target"):
        self.min_value = df[target].min()
        self.max_value = df[target].max()
        self.model.fit(df[["reg_pred", "bin_pred"]], df[target])
        
    def predict(self, df):
        pred = self.model.predict(df[["reg_pred", "bin_pred"]])
        return np.clip(pred, self.min_value, self.max_value)
        