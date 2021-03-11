def zigzag(n):
    '''zigzag rows'''
    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)
    xs = list(range(n))
    return {index: n for n, index in enumerate(sorted(
        ((x, y) for x in xs for y in xs),
        key=compare
    ))}


def printzz(myarray):
    '''show zigzag rows as lines'''
    n = int(len(myarray) ** 0.5 + 0.5)
    xs = list(range(n))
    print(('\n'.join(
        [''.join("%3i" % myarray[(x, y)] for x in xs) for y in xs]
    )))


printzz(zigzag(6))
