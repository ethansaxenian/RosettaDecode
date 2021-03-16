

def accumulator(sum):
    def f(n):
        f.sum += n
        return f.sum

    f.sum = sum
    return f

x = accumulator(1)
x(5)
x(2.3)
x = accumulator(1)
x(5)
x(2.3)
x2 = accumulator(3)
x2(5)
x2(3.3)
x(0)
x2(0)
