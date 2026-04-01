import numpy as np

def _sigmoid(z):
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=500):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float).reshape(-1, 1)
    
    N, D = X.shape
    
    w = np.zeros((D, 1))
    b = 0.0
    
    for _ in range(steps):
        z = X @ w + b              # (N,1)
        p = _sigmoid(z)            # (N,1)
        
        error = p - y              # (N,1)
        
        dw = (X.T @ error) / N     # (D,1)
        db = float(np.mean(error)) # scalar
        
        w -= lr * dw
        b -= lr * db
    
    # 👇 FIX QUAN TRỌNG: reshape output
    return w.flatten(), float(b)