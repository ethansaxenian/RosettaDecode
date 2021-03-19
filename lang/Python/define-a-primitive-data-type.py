class num(int):
    def __init__(self, b):
        if 1 <= b <= 10:
            return int.__init__(self+0)
        else:
            raise ValueError("Value %s should be >=0 and <= 10" % b)


x = num(3)
x = num(11)


x
3
type(x)

