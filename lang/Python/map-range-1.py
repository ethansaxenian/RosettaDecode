def maprange( a, b, s):
	(a1, a2), (b1, b2) = a, b
	return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))

for s in range(11):
	print("%2g maps to %g" % (s, maprange( (0, 10), (-1, 0), s)))


