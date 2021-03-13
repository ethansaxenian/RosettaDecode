def popcount(n): return bin(n).count("1")

primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}
p, i = [], 0
while len(p) < 25:
        if popcount(i) in primes: p.append(i)
        i += 1


p
[3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 31, 33, 34, 35, 36]
p, i = [], 888888877
while i <= 888888888:
        if popcount(i) in primes: p.append(i)
        i += 1


p
[888888877, 888888878, 888888880, 888888883, 888888885, 888888886]

