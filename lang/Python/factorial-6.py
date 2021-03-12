from numpy import prod

def factorial(n):
    return prod(list(range(1, n + 1)), dtype=int)
