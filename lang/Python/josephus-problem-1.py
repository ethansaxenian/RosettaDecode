def j(n, k):
	p, i, seq = list(range(n)), 0, []
	while p:
		i = (i+k-1) % len(p)
		seq.append(p.pop(i))
	return 'Prisoner killing order: %s.\nSurvivor: %i' % (', '.join(str(i) for i in seq[:-1]), seq[-1])

print(j(5, 2))
print(j(41, 3))

