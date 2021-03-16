for n in range(34):
    print(" {0:6b} {1:3o} {2:2d} {3:2X}".format(n, n, n, n))
# The following would give the same output, and,
# due to the outer brackets, works with Python 3.0 too
# print( " {n:6b} {n:3o} {n:2d} {n:2X}".format(n=n) )
