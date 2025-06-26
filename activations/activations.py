"""Code for various activation functions used in LLMs."""
import numpy as np
import scipy

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Classic piece-wise linear
# ReLU, PReLU, Square ReLU

def ReLU(x: np.ndarray):
    """ReLU: Rectified Linear Unit."""
    return np.maximum(0, x)

def PReLU(x: np.ndarray, alpha=0.25):
    """Leaky ReLU: Parameteric Rectified Linear Unit."""
    return np.where(x >= 0, x, alpha * x)

def SqReLU(x: np.ndarray):
    """Square ReLU: """
    return np.square(np.maximum(0, x))

# Smooth “Gaussian-like” nonlinearities
def GeLU(x: np.ndarray):
    """Gaussian Error Linear Unit. erf falls in [-1, 1] range."""
    return 0.5 * x * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))

def QuickGeLU(x: np.ndarray):
    """QuickGELU (Hendrycks and Gimpel approx). """
    return x / (1.0 + np.exp(-1.702 * x))

# SiLU
def Swish(x: np.ndarray):
    """Sigmoid Linear Unit."""
    return x / (1 + np.exp(-x))

def Mish(x: np.ndarray):
    return x * np.tanh(np.log1p(np.exp(x)))

# Gated Linear Unit (GLU) family
def GLU(x: np.ndarray):
    return x * sigmoid(x)

def GeGLU(x: np.ndarray):
    return x * GeLU(x)

def SwiGLU(x: np.ndarray):
    return x * Swish(x)
