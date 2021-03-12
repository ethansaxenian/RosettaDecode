def square(n):
    return n * n


numbers = [1, 3, 5, 7]

squares1 = [square(n) for n in numbers]  # list comprehension

squares2a = list(map(square, numbers))  # functional form

squares2b = [x * x for x in numbers]  # functional form with `lambda`

squares3 = [n * n for n in numbers]  # no need for a function,
# anonymous or otherwise

isquares1 = (n * n for n in numbers)  # iterator, lazy

import itertools

isquares2 = map(square, numbers)  # iterator, lazy
