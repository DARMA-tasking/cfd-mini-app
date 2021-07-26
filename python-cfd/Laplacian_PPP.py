
def cartesian_to_index(i, j, n):
    return j * n + i

def fill(m, n):
    # Iterate row major
    for j in range(0, n):
        for i in range(0, n):
            k = cartesian_to_index(i, j, n)
            # Initialize diagonal value
            v = - 4

            # Detect corners and borders
            if not i:
                v+= 1
            if i == n - 1:
                v+= 1
            if not j:
                v+= 1
            if j == n - 1:
                v+= 1

            # Assign diagonal entry
            m[k][k] = v

            # Assign unit entries
            if i:
                m[k][k-1] = 1
            if i < n - 1:
                m[k][k+1] = 1
            if j:
                m[k][k-n] = 1
            if j < n - 1:
                m[k][k+n] = 1
    return m
