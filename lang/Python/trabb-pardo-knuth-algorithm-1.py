
def f(x): return abs(x) ** 0.5 + 5 * x**3

print(', '.join('%s:%s' % (x, v if v<=400 else "TOO LARGE!")
	           for x,v in ((y, f(float(y))) for y in input('\nnumbers: ').strip().split()[:11][::-1])))

