import numpy as np
import control as ct
import warnings

def calculate_poles(sys):
    """
    Calculate the poles of a multivariable linear dynamic system.

    Args:
        sys (control.StateSpace or control.TransferFunction): The system.

    Returns:
        np.ndarray: Array of poles.
    """
    return ct.poles(sys)

def calculate_zeros(sys):
    """
    Calculate the zeros of a multivariable linear dynamic system.

    Args:
        sys (control.StateSpace or control.TransferFunction): The system.

    Returns:
        np.ndarray: Array of zeros.
    """
    return ct.zeros(sys)

def calculate_singular_values(sys, omega=0):
    """
    Calculate the singular values of the system frequency response at a given frequency.

    Args:
        sys (control.StateSpace or control.TransferFunction): The system.
        omega (float): Frequency in rad/s. Default is 0 (steady state).

    Returns:
        np.ndarray: Array of singular values, sorted in descending order.
    """
    G_jw = ct.evalfr(sys, omega * 1j)
    # Ensure G_jw is a 2D matrix even if scalar
    if np.isscalar(G_jw):
        G_jw = np.array([[G_jw]])
    elif G_jw.ndim == 0:
        G_jw = np.array([[G_jw]])

    # Singular Value Decomposition
    U, S, Vh = np.linalg.svd(G_jw)
    return S

def relative_gain_array(G):
    """
    Calculate the Relative Gain Array (RGA) for a given gain matrix G.
    RGA(G) = G .* (G^-1)^T

    Args:
        G (np.ndarray): The gain matrix (e.g. steady state gain).

    Returns:
        np.ndarray: The RGA matrix.
    """
    try:
        G_inv = np.linalg.inv(G)
        RGA = G * G_inv.T
        return RGA
    except np.linalg.LinAlgError:
        warnings.warn("Matrix is singular, RGA cannot be computed.")
        return None

def system_gain(sys, omega=0):
    """
    Calculate the system gain matrix at a given frequency.

    Args:
        sys (control.StateSpace or control.TransferFunction): The system.
        omega (float): Frequency in rad/s.

    Returns:
        np.ndarray: The frequency response matrix at the given frequency.
    """
    return ct.evalfr(sys, omega * 1j)
