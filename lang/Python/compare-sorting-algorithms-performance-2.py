def ones(n):
    return [1]*n

def reversedrange(n):
    return reversed(list(range(n)))

def shuffledrange(n):
    x = list(range(n))
    random.shuffle(x)
    return x
