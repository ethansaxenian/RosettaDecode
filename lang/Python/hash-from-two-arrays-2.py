keys = ['a', 'b', 'c']
values = [1, 2, 3]
hash = dict(list(zip(keys, values)))

# Lazily, Python 2.3+, not 3.x:

hash = dict(zip(keys, values))
