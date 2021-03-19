from toolz import curry
import operator
add = curry(operator.add)
add2 = add(2)
add2
add2(7)
9
# Toolz also has pre-curried versions of most HOFs from builtins, stdlib, and toolz
from toolz.curried import map
double = map(lambda x: x*2)
print(*double(range(5)))
