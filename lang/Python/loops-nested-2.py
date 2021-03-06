from random import randint

class Found20(Exception):
    pass

mat = [[randint(1, 20) for x in range(10)] for y in range(10)]

try:
    for row in mat:
        for item in row:
            print(item, end=' ')
            if item == 20:
                raise Found20
        print()
except Found20:
    print()
