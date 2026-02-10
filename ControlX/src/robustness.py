import numpy as np
import control as ct
import warnings

def sensitivity_function(G, K):
    """
    Calculate the sensitivity function S(s) = (I + G(s)K(s))^-1.

    Args:
        G (control.StateSpace or control.TransferFunction): The plant.
        K (control.StateSpace or control.TransferFunction): The controller.

    Returns:
        control.StateSpace: The sensitivity function S.
    """
    L = G * K
    # Sensitivity Function S = (I + L)^-1
    # control.feedback returns L / (1+L) if sign=-1
    # To get (1+L)^-1, we can compute 1 - T
    # Or simply feedback(1, L, sign=-1)

    # Using formula: S = (I + G*K)^-1
    # We can use feedback(I, G*K) ? No.
    # feedback(sys1, sys2) computes sys1 / (1 + sys1*sys2)
    # S = feedback(1, G*K) assuming identity feedback path?
    # If sys1 is identity (size of outputs of L), and sys2 is L.

    # Correct way using control library:
    # S = feedback(I, L) where I is identity with size equal to number of outputs

    # However, if G and K are MIMO, we need to be careful with dimensions.
    # Let's assume standard negative feedback.

    # Try using feedback(eye(n_outputs), L)

    n_outputs = G.noutputs
    I = ct.ss([], [], [], np.eye(n_outputs))
    S = ct.feedback(I, L)
    return S

def complementary_sensitivity_function(G, K):
    """
    Calculate the complementary sensitivity function T(s) = G(s)K(s)(I + G(s)K(s))^-1.
    T = I - S

    Args:
        G (control.StateSpace or control.TransferFunction): The plant.
        K (control.StateSpace or control.TransferFunction): The controller.

    Returns:
        control.StateSpace: The complementary sensitivity function T.
    """
    L = G * K
    # T = L / (1 + L)
    # Using feedback(L, I) or feedback(L, 1) if SISO
    n_inputs = L.ninputs
    I = ct.ss([], [], [], np.eye(n_inputs))
    T = ct.feedback(L, I)
    return T

def calculate_hinf_norm(sys, omega=None):
    """
    Calculate the H-infinity norm of a system by sampling frequency response.

    Args:
        sys (control.StateSpace or control.TransferFunction): The system.
        omega (array-like, optional): Frequency points. If None, generated automatically.

    Returns:
        float: The approximated H-infinity norm.
    """
    if omega is None:
        omega = np.logspace(-2, 2, 1000)

    # Calculate frequency response
    # control.freqresp returns mag, phase, omega
    # But for MIMO, it returns complex response if return_xfer=False?
    # No, control.freqresp returns mag, phase, omega.
    # We want singular values.

    # evalfr returns complex response at one frequency.
    # freqresp can return (mag, phase, omega) but mag is singular values only for SISO?
    # For MIMO, let's just loop or use singular_values from analysis.py if imported,
    # or just compute here.

    max_sv = 0
    for w in omega:
        resp = ct.evalfr(sys, w*1j)
        if np.isscalar(resp):
            sv = np.abs(resp)
        else:
            sv = np.max(np.linalg.svd(resp, compute_uv=False))
        if sv > max_sv:
            max_sv = sv

    return max_sv

def small_gain_theorem_check(M, Delta, omega=None):
    """
    Check stability using the Small Gain Theorem.
    Specifically, check if ||M||_inf * ||Delta||_inf < 1.

    Args:
        M (control.StateSpace): The nominal closed-loop system seen by the uncertainty.
        Delta (control.StateSpace or float): The uncertainty.
        omega (array-like, optional): Frequency points for norm approximation.

    Returns:
        bool: True if stable, False otherwise.
        float: The product of norms.
    """
    if isinstance(M, (ct.StateSpace, ct.TransferFunction)):
        norm_M = calculate_hinf_norm(M, omega)
    else:
        norm_M = np.linalg.norm(M, 2) # Assume matrix gain

    if isinstance(Delta, (ct.StateSpace, ct.TransferFunction)):
        norm_Delta = calculate_hinf_norm(Delta, omega)
    else:
        norm_Delta = np.abs(Delta)

    product = norm_M * norm_Delta
    return product < 1.0, product

def robust_stability_margin(S, omega=None):
    """
    Calculate the robust stability margin, which is 1 / ||T||_inf for multiplicative uncertainty.

    Args:
        S (control.StateSpace): Sensitivity or Complementary Sensitivity function.
        omega (array-like, optional): Frequency points for norm approximation.

    Returns:
        float: The stability margin.
    """
    norm_S = calculate_hinf_norm(S, omega)
    if norm_S == 0:
        return float('inf')
    return 1.0 / norm_S
