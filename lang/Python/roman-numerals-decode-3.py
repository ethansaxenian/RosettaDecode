from functools import reduce
numerals = { 'M' : 1000, 'D' : 500, 'C' : 100, 'L' : 50, 'X' : 10, 'V' : 5, 'I' : 1 }
def romannumeral2number(s):
        return reduce(lambda x, y: -x + y if x < y else x + y, [numerals.get(x, 0) for x in s.upper()])
