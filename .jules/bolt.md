## 2024-05-19 - Vectorize `calculate_singular_values` for Multiple Frequencies
**Learning:** `control.StateSpace.evalfr` is extremely slow when used in a `for` loop over many frequencies for computing singular values (such as creating Sigma plots). In Python, `control` can compute frequency responses natively for an array of frequencies using `sys.frequency_response(omega).complex`, which returns a `(outputs, inputs, frequencies)` array (for MIMO systems). This can be fed directly to a vectorized `numpy.linalg.svd` by transposing to `(frequencies, outputs, inputs)` for significant speedup over looping.
**Action:** When computing any metric over a frequency array, prefer `sys.frequency_response(omega)` and vectorized `numpy` operations instead of python loops with `evalfr`.

## 2025-03-28 - Vectorizing CVXPY Constraints
**Learning:** In MPC problems using CVXPY, Python loops for constraints (e.g. `for k in range(N): constraints += [x[:, k+1] == A@x[:, k] + B@u[:, k]]`) create a significant overhead during both compilation and solve time.
**Action:** Always vectorize state dynamics constraints using matrix slicing (e.g., `_x[:, 1:] == A @ _x[:, :-1] + B @ _u`) which provides ~3.5x faster problem setup and ~2.5x faster solve times.
