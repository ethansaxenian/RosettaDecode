def perf2(n):
    return n == sum(i for i in range(1, n) if n % i == 0)

print((
    list(filter(perf2, list(range(1, 10001))))
))
