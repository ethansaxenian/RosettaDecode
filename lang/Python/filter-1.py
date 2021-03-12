values = list(range(10))
evens = [x for x in values if not x & 1]
ievens = (x for x in values if not x & 1) # lazy
# alternately but less idiomatic:
evens = [x for x in values if not x & 1]
