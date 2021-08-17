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
rankA = np.linalg.matrix_rank(A)
print("  rank:", rankA)

# Verify that problem is well-posed when nullspace has dimension 1
if rankA == len(b) - 1:
    kerA = [a[0] for a in sp.linalg.null_space(A)]
    print("  nullspace: <", kerA, '>')
    print("  kerA . b:", kerA @ b)

##### Actual solver starts here #####

# Initialize containers and values
print("# intialization:")
x = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
print("  initial guess:", x)

residual = b - A @ x
rms2 = residual @ residual
print("  RMS error:", math.sqrt(rms2))

direction = residual
print("  new direction:", direction)

# Iterate
for k in range(min(A.shape)):
    kp1 = k + 1
    print("# iteration:", kp1)

    # Compute step
    alpha = rms2 / (direction @ A @ direction)
    print("  alpha[{}]: {}".format(k, alpha))

    # Update solution
    x += alpha * direction
    print("  updated solution:", x)

    # Update residual
    residual = b - A @ x
    new_rms2 = residual @ residual
    print("  RMS error:", math.sqrt(new_rms2))

    # Terminate early when possible
    if new_rms2 < 1e-6:
        print("# CG converged to solution: {} with residual norm: {}".format(
            x, math.sqrt(new_rms2)))
        sys.exit(0)

    # Compute new direction
    beta = new_rms2 / rms2
    print("  beta[{}]: {}".format(k, beta))
    direction = residual - beta * direction
    print("  new direction:", direction)

    # Update residual squared L2
    rms2 = new_rms2
