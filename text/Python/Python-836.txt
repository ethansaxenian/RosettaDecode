def triplets(n):
    for x in range(1, n + 1):
        for y in range(x, n + 1):
            for z in range(y, n + 1):
                yield x, y, z
