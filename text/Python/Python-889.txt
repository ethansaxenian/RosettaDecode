from random import randint

mat = [[randint(1, 20) for x in range(10)] for y in range(10)]

found20 = False
for row in mat:
    for item in row:
        print(item, end=' ')
        if item == 20:
            found20 = True
            break
    print()
    if found20:
        break
