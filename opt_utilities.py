import numpy as np

def compute_radius_lb(N, eta, c):
    """
    Computes the concentration radius epsilon_N(eta) lower bound

    Parameters:
    - N: int, number of samples
    - eta: float, confidence parameter (0 < eta < 1)
    - c: float, constant from the theorem (c > 1)

    Returns:
    - epsilon: float, the concentration bound
    """
    assert 0 < eta <= 1, "eta must be in (0, 1]"
    assert c > 1, "c must be greater than 1"
    return np.log(c / eta) / np.sqrt(N)