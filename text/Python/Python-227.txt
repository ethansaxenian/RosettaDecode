

from pprint import pprint
from math import sqrt


def cholesky(A):
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            L[i][j] = sqrt(A[i][i] - s) if (i == j) else \
                      (1.0 / L[j][j] * (A[i][j] - s))
    return L

if __name__ == "__main__":
    m1 = [[25, 15, -5],
          [15, 18,  0],
          [-5,  0, 11]]
    pprint(cholesky(m1))
    print()

    m2 = [[18, 22,  54,  42],
          [22, 70,  86,  62],
          [54, 86, 174, 134],
          [42, 62, 134, 106]]
    pprint(cholesky(m2), width=120)
