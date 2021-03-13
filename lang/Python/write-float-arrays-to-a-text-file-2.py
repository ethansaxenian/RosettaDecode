import math
x = [1, 2, 3, 1e11]
y = map(math.sqrt, x)
y
[1.0, 1.4142135623730951, 1.7320508075688772, 316227.76601683791]
writedat("sqrt.dat", x, y)
# check

for line in open('sqrt.dat'):
   print(line,)
