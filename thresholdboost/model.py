import numpy as np
from scipy.optimize import minimize
from .utils import sigmoid, log_odds, soft_threshold_objective

class ThresholdUnit:
    def __init__(self, lambda_reg=0.01):
        self.lambda_reg = lambda_reg
        self.feature_idx = None
        self.w = None
        self.c = None
        self.gamma = None

    def fit(self, X, r):
        n_features = X.shape[1]
        best_loss = float('inf')
        best_params = None
        best_feature = None

        for j in range(n_features):
            xj = X[:, j]
            c_init = np.median(xj)
            w_init = 1.0 / (np.std(xj) + 1e-5)
            gamma_init = np.std(r) + 0.01 
            
            init_params = np.array([w_init, c_init, gamma_init])
            bounds = [(-100.0, 100.0), (np.min(xj), np.max(xj)), (None, None)]

            res = minimize(
                soft_threshold_objective, init_params, 
                args=(xj, r, self.lambda_reg),
                method='L-BFGS-B', bounds=bounds
            )

            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
                best_feature = j

        self.feature_idx = best_feature
        self.w, self.c, self.gamma = best_params

    def predict(self, X):
        xj = X[:, self.feature_idx]
        return self.gamma * sigmoid(self.w * (xj - self.c))

class ThresholdBoost:
    def __init__(self, n_estimators=50, learning_rate=0.1, lambda_reg=0.01):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.estimators = []
        self.base_score = 0.0

    def fit(self, X, y):
        y = np.array(y).astype(float)
        p_initial = np.mean(y)
        self.base_score = log_odds(p_initial)
        F = np.full(y.shape, self.base_score)

        for i in range(self.n_estimators):
            p_hat = sigmoid(F)
            pseudo_residuals = y - p_hat
            
            unit = ThresholdUnit(lambda_reg=self.lambda_reg)
            unit.fit(X, pseudo_residuals)
            self.estimators.append(unit)
            F += self.learning_rate * unit.predict(X)

        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.base_score)
        for unit in self.estimators:
            F += self.learning_rate * unit.predict(X)
        probs = sigmoid(F)
        return np.vstack((1 - probs, probs)).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)