import numpy as np

class MyMinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        X = np.array(X)
        self.min_, self.max_ = X.min(axis=0), X.max(axis=0)
        return self
    
    def transform(self, X):
        X = np.array(X)
        return (X - self.min_) / (self.max_ - self.min_ + 1e-9)


class MyStandartScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.array(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    
    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / (self.std_ + 1e-9)
    

def r2_score_custom(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (numerator /denominator) 
