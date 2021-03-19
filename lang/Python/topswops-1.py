from itertools import permutations
def f1(p):
	i = 0
	while True:
		p0  = p[0]
		if p0 == 1: break
		p[:p0] = p[:p0][::-1]
		i  += 1
	return i

def fannkuch(n):
	return max(f1(list(p)) for p in permutations(range(1, n+1)))

for n in range(1, 11): print(n,fannkuch(n))


