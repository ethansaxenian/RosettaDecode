width = int(input("Width of myarray: "))
height = int(input("Height of Array: "))
myarray = [[0] * width for i in range(height)]
myarray[0][0] = 3.5
print((myarray[0][0]))
