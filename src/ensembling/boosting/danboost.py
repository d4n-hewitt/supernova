import numpy as np


class DanBoost:
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []

    def fit_estimator(self, X, y):
        for _ in range(self.n_estimators):
            model = self.base_estimator(input_shape=X.shape[1], output_shape=1)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict_latest_estimator(self, X):
        model = self.models[-1]
        predictions = model.predict(X)
        return predictions

    def calc_error(self, X):
        predictions = self.predict_latest_estimator(self, X)
        errors = np.abs(predictions - X)
        return errors
