def arithmetic(x, y):
    for op in "+ - * // % **".split():
        expr = "%(x)s %(op)s %(y)s" % vars()
        print(("%s\t=> %s" % (expr, eval(expr))))


arithmetic(12, 8)
arithmetic(eval(input("Number 1: ")), eval(input("Number 2: ")))
