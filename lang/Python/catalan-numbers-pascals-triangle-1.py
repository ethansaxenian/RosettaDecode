n = 15
t = [0] * (n + 2)
t[1] = 1
for i in range(1, n + 1):
	for j in range(i, 1, -1): t[j] += t[j - 1]
	t[i + 1] = t[i]
	for j in range(i + 1, 1, -1): t[j] += t[j - 1]
	print(t[i+1] - t[i], end=' ')

	
1 2 5 14 42 132 429 1430 4862 16796 58786 208012 742900 2674440 9694845

