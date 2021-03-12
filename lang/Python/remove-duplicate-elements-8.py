# nubBy :: (a -> a -> Bool) -> [a] -> [a]
def nubBy(p, xs):
    def go(xs):
        if xs:
            x = xs[0]
            return [x] + go(
                list([y for y in xs[1:] if not p(x, y)])
            )
        else:
            return []
    return go(xs)
