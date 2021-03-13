def fiblike(start):
	addnum = len(start)
	memo = start[:]
	def fibber(n):
		try:
			return memo[n]
		except IndexError:
			ans = sum(fibber(i) for i in range(n-addnum, n))
			memo.append(ans)
			return ans
	return fibber

fibo = fiblike([1,1])
[fibo(i) for i in range(10)]
[1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
lucas = fiblike([2,1])
[lucas(i) for i in range(10)]
[2, 1, 3, 4, 7, 11, 18, 29, 47, 76]
for n, name in zip(range(2,11), 'fibo tribo tetra penta hexa hepta octo nona deca'.split()) :
	fibber = fiblike([1] + [2**i for i in range(n-1)])
	print('n=%2i, %5snacci -> %s ...' % (n, name, ' '.join(str(fibber(i)) for i in range(15))))

