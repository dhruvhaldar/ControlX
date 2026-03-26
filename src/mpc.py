import numpy as np
import cvxpy as cp
import control as ct
import scipy.sparse as sp

class MPCController:
    """
    Model Predictive Controller.
    """
    def __init__(self, sys, Q, R, N, dt=0.1, constraints=None):
        """
        Initialize the MPC controller.

        Args:
            sys (control.StateSpace): The system (continuous or discrete).
            Q (np.ndarray): State weighting matrix.
            R (np.ndarray): Input weighting matrix.
            N (int): Prediction horizon.
            dt (float): Sampling time.
            constraints (dict): Dictionary of constraints.
                'xmin': np.ndarray or float
                'xmax': np.ndarray or float
                'umin': np.ndarray or float
                'umax': np.ndarray or float
        """
        # Security: Input validation to prevent resource exhaustion
        if not isinstance(N, int) or N <= 0 or N > 1000:
            raise ValueError("Prediction horizon N must be a positive integer <= 1000")
        if dt <= 0:
            raise ValueError("Sampling time dt must be positive")

        self.dt = dt
        self.N = N
        self.Q = Q
        self.R = R
        self.constraints = constraints if constraints else {}

        # Discretize system if continuous
        if sys.dt is None or sys.dt == 0:
            self.sys_d = ct.c2d(sys, dt)
        else:
            self.sys_d = sys

        self.A = self.sys_d.A
        self.B = self.sys_d.B
        self.n_x = self.sys_d.nstates
        self.n_u = self.sys_d.ninputs

        # Compute terminal cost P using DARE (optional, often P=Q is used or solution to Riccati)
        # Solve P = A'PA - A'PB(R + B'PB)^-1 B'PA + Q
        try:
            # control.dare solves: X = A'XA - A'XB(R + B'XB)^-1 B'XA + Q
            # Returns X, L, G
            X, _, _ = ct.dare(self.A, self.B, self.Q, self.R)
            self.P = X
        except Exception:
            # Security: Do not leak exception details in console
            print("Warning: Could not compute terminal cost P. Using Q.")
            self.P = self.Q

        # Setup parameterized problem for performance
        self._setup_problem()

    def _setup_problem(self):
        """
        Set up the parameterized CVXPY problem to avoid recompilation at each step.
        """
        self._x = cp.Variable((self.n_x, self.N + 1))
        self._u = cp.Variable((self.n_u, self.N))
        self._x0_param = cp.Parameter(self.n_x)

        cost = 0
        constraints = [self._x[:, 0] == self._x0_param]

        # ⚡ Bolt Optimization: Fully vectorize MPC problem formulation
        # Replaces O(N) loop with vectorized operations for state, input, and cost computations.
        # This significantly speeds up problem setup and solving via CVXPY.

        # Block diagonal matrices for vectorized cost computation
        Q_big = sp.block_diag([self.Q] * self.N)
        R_big = sp.block_diag([self.R] * self.N)

        import warnings
        # Vectorized cost
        # Suppress FutureWarnings about `order` since we explicitly rely on default Fortran ordering for compatibility
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            cost = cp.quad_form(cp.vec(self._x[:, :-1]), Q_big) + cp.quad_form(cp.vec(self._u), R_big)
        cost += cp.quad_form(self._x[:, self.N], self.P) # Terminal cost

        # Vectorized state dynamics constraint
        constraints += [self._x[:, 1:] == self.A @ self._x[:, :-1] + self.B @ self._u]

        # Vectorized input constraints
        if 'umin' in self.constraints:
            umin = np.atleast_1d(self.constraints['umin']).reshape(-1, 1) if isinstance(self.constraints['umin'], (np.ndarray, list)) else self.constraints['umin']
            constraints += [self._u >= umin]
        if 'umax' in self.constraints:
            umax = np.atleast_1d(self.constraints['umax']).reshape(-1, 1) if isinstance(self.constraints['umax'], (np.ndarray, list)) else self.constraints['umax']
            constraints += [self._u <= umax]

        # Vectorized state constraints
        if 'xmin' in self.constraints:
            xmin = np.atleast_1d(self.constraints['xmin']).reshape(-1, 1) if isinstance(self.constraints['xmin'], (np.ndarray, list)) else self.constraints['xmin']
            constraints += [self._x[:, 1:] >= xmin]
        if 'xmax' in self.constraints:
            xmax = np.atleast_1d(self.constraints['xmax']).reshape(-1, 1) if isinstance(self.constraints['xmax'], (np.ndarray, list)) else self.constraints['xmax']
            constraints += [self._x[:, 1:] <= xmax]

        self._prob = cp.Problem(cp.Minimize(cost), constraints)

    def compute_control(self, x0):
        """
        Compute the optimal control input for the current state x0.

        Args:
            x0 (np.ndarray): Current state vector.

        Returns:
            u0 (np.ndarray): Optimal control input to apply.
            status (str): Solver status.
        """
        # Security: Input validation to prevent solver crashes or exceptions
        try:
            x0_arr = np.array(x0, dtype=float)
        except (ValueError, TypeError):
            print("MPC Error: Input state must be a valid numeric array or sequence.")
            return np.zeros(self.n_u), "invalid_input"

        if x0_arr.shape != (self.n_x,) and x0_arr.shape != (self.n_x, 1):
            print(f"MPC Error: Invalid state dimension. Expected {self.n_x}, got {x0_arr.shape}")
            return np.zeros(self.n_u), "invalid_dimension"

        if not np.isfinite(x0_arr).all():
            print("MPC Error: Input state contains NaN or infinite values.")
            return np.zeros(self.n_u), "invalid_values"

        # Set the current state parameter
        self._x0_param.value = x0_arr.flatten()

        # Solve
        self._prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if self._prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"MPC Warning: Solver status: {self._prob.status}")
            return np.zeros(self.n_u), self._prob.status

        return self._u[:, 0].value, self._prob.status
