import numpy as np

class MyLinearRegression:
    def __init__(self, eta=0.01, n_iter=1000, random_state=21):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        X_array, y_array = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
        
        for _ in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_array[indices]
            y_shuffled = y_array[indices]

            for xi, target in zip(X_shuffled, y_shuffled):
                output = np.dot(xi, self.w) + self.b
                error = target - output
                self.w += self.eta * 2 * error * xi
                self.b += self.eta * 2 * error

    def predict(self, X):
        return np.dot(X, self.w) + self.b 
    

class MyRegularizedRegression:
    def __init__(self, eta=0.001, n_iter=100, alpha=1.0, l1_ratio=0.5, random_state=21):
        self.eta = eta
        self.n_iter = n_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.w, self.b = None, None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shpae
        self.w = np.zeros(n_features)
        self.b = 0.0

        X_array, y_array = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

        for _ in range(self.n_iter):
            indices = np.random.permutation(n_samples)

            for xi, target in zip(X_array[indices], y_array[indices]):

                prediction = np.dot(xi, self.w) + self.b
                error = target - prediction

                l1_grad = self.alpha * self.l1_ratio * np.sign(self.w)
                
                l2_grad = self.alpha * (1 - self.l1_ratio) * 2 * self.w

                self.w += self.eta * (2 * error * xi - l1_grad - l2_grad)
                self.b += self.eta * 2 * error

    def predict(self, X):
        X_array = np.array(X, np.float64)
        return np.dot(X_array, self.w) + self.b 
    

class MyLogisticRegressionSGD:
    def __init__(self, lr=0.01, epochs=10, batch_size=256,random_state=42):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.w, self.b = None, None

    def _sigmoid(self, z):
        z = np.clip(z, -25, 25)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X, y = np.array(X, dtype=np.float64), np.array(y)
        n_samples, n_features = X.shape
        self.w, self.b = np.zeros(n_features), 0.0
        np.random.seed(self.random_state)
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size] 
                y_batch = y_shuffled[i : i + self.batch_size]

                z = np.dot(X_batch, self.w) + self.b 
                y_pred = self._sigmoid(z) 

                error = y_pred - y_batch

                dw = np.dot(X_batch.T, error) / len(y_batch)
                db = np.mean(error)

                self.w -= self.lr * dw
                self.b -= self.lr * db

        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        z = np.dot(X, self.w) + self.b

        prob_class_1 = self._sigmoid(z)
        return np.column_stack([1 - prob_class_1, prob_class_1])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int) 