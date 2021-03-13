from itertools import compress, cycle
def josephus(prisoner, kill, surviver):
    p = range(prisoner)
    k = [0] * kill
    k[kill-1] = 1
    s = [1] * kill
    s[kill -1] = 0
    queue = p

    queue = compress(queue, cycle(s))
    try:
        while True:
            p.append(queue.next())
    except StopIteration:
        pass

    kil=[]
    killed = compress(p, cycle(k))
    try:
        while True:
            kil.append(killed.next())
    except StopIteration:
        pass

    print('The surviver is: ', kil[-surviver:])
    print('The kill sequence is ', kil[:prisoner-surviver])

