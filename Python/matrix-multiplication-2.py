from operator import mul

def matrixMul(m1, m2):
  return [list(map(
        lambda *column:
          sum(map(mul, row, column)),
        *m2)) for row in m1]
