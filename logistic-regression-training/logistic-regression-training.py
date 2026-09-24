import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    # Number of training examples and features

    m, n = X.shape

    # Initialize weights and bias

    w = np.zeros(n)

    b = 0.0

    # Gradient descent

    for _ in range(steps):

        # 1. Linear score

        z = X @ w + b

        # 2. Convert scores to probabilities

        p = _sigmoid(z)

        # 3. Prediction error

        error = p - y

        # 4. Gradients

        dw = (X.T @ error) / m

        db = np.mean(error)

        # 5. Update parameters

        w -= lr * dw

        b -= lr * db

    return w, b