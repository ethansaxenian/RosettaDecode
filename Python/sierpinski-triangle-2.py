import functools

def sierpinski(n):

    def aggregate(TRIANGLE, I):
        SPACE = " " * (2 ** I)
        return [SPACE+X+SPACE for X in TRIANGLE] + [X+" "+X for X in TRIANGLE]

    return functools.reduce(aggregate, list(range(n)), ["*"])

print(("\n".join(sierpinski(4))))
