def in_order(l):
    return all( l[i] <= l[i+1] for i in range(0,len(l)-1))
