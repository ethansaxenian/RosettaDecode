from operator import mul
from functools import reduce

def factorial(n):
    return reduce(mul, list(range(1,n+1)), 1)