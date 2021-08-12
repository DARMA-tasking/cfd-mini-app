import sys
import math
import numpy as np
import scipy as sp
from scipy import linalg

A = np.array([[-2,1,1,0],[1,-2,0,1],[1,0,-2,1],[0,1,1,-2]])
b = np.array([1, -1, 1, -1])

# Provide some information
print("# matrix:")
for r in A:
    print("   ", r)
print("  eigenvalues:", ", ".join(["{:.6g}".format(e) for e in np.linalg.eigvals(A)]))
print("  rank:", np.linalg.matrix_rank(A))

kerA = sp.linalg.null_space(A) 
print("  nullspace:")
for r in kerA:
    print("   ", r)

# Initialize containers and values
print("# intialization:")
x = np.array([0., 0., 0., 0.])
print("  initial guess:", x)

residuals = [b - A @ x]
rms2 = residuals[0] @ residuals[0]
print("  residual:{} (L2 norm: {})".format(residuals[0], rms2))

directions = residuals[:]
print("  directions", [list(d) for d in directions])
alpha, beta = [], []

# Iterate
for k in range(min(A.shape)):
    kp1 = k + 1
    print("# iteration:", kp1)

    # Compute step
    alpha.append(rms2 / (directions[k] @ A @ directions[k]))
    print("  alpha[{}]: {}".format(k, alpha[k]))

    # Update solution
    x += alpha[k] * directions[k]
    print("  updated solution:", x)

    # Update residual
    residuals.append(b - A @ x)
    new_rms2 = residuals[kp1] @ residuals[kp1]
    print("  residual:{} (L2 norm: {})".format(residuals[kp1], new_rms2))

    # Terminate early when possible
    if new_rms2 < 1e-12:
        print("# CG converged to solution: {} with residual norm: {}".format(
            x, math.sqrt(new_rms2)))
        sys.exit(0)

    # Compute new direction
    beta.append(new_rms2 / rms2)
    print("  beta[{}]: {}".format(k, beta[k]))
    directions.append(residuals[kp1] - beta[k] * directions[k])
    print("  directions", [list(d) for d in directions])

    # Update residual squared L2
    rms2 = new_rms2
