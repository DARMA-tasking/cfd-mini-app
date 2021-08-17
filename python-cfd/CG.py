import sys
import math
import numpy as np
import scipy as sp
from scipy import linalg

A = np.array([
    [-2,1,0,1,0,0,0,0,0],
    [1,-3,1,0,1,0,0,0,0],
    [0,1,-2,0,0,1,0,0,0],
    [1,0,0,-3,1,0,1,0,0],
    [0,1,0,1,-4,1,0,1,0],
    [0,0,1,0,1,-3,0,0,1],
    [0,0,0,1,0,0,-2,1,0],
    [0,0,0,0,1,0,1,-3,1],
    [0,0,0,0,0,1,0,1,-2]])
b = np.array([1, 1, 1, 1, 0, -1, -1, -1, -1])

# Provide some information: not necessary to the method
print("# matrix:")
for r in A:
    print("   ", r)
print("  eigenvalues:", ", ".join(["{:.6g}".format(e) for e in np.linalg.eigvals(A)]))
print("  rank:", np.linalg.matrix_rank(A))

kerA = sp.linalg.null_space(A)
print("  nullspace:")
for r in kerA:
    print("   ", r)


##### Actual solver starts here #####

# Initialize containers and values
print("# intialization:")
x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
print("  initial guess:", x)

residual = b - A @ x
rms2 = residual @ residual
print("  RMS error:", math.sqrt(rms2))

directions = [residual]
print("  directions", [list(d) for d in directions])

# Iterate
for k in range(min(A.shape)):
    kp1 = k + 1
    print("# iteration:", kp1)

    # Compute step
    alpha = rms2 / (directions[k] @ A @ directions[k])
    print("  alpha[{}]: {}".format(k, alpha))

    # Update solution
    x += alpha * directions[k]
    print("  updated solution:", x)

    # Update residual
    residual = b - A @ x
    new_rms2 = residual @ residual
    print("  RMS error:", math.sqrt(new_rms2))

    # Terminate early when possible
    if new_rms2 < 1e-8:
        print("# CG converged to solution: {} with residual norm: {}".format(
            x, math.sqrt(new_rms2)))
        sys.exit(0)

    # Compute new direction
    beta = new_rms2 / rms2
    print("  beta[{}]: {}".format(k, beta))
    directions.append(residual - beta * directions[k])
    print("  new direction:", directions[-1])

    # Update residual squared L2
    rms2 = new_rms2
