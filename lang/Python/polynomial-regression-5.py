import numpy

p = numpy.poly1d(numpy.polyfit(x, y, deg=2), variable='N')
print(p)
