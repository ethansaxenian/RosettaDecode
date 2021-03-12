((x,y,z) for x in range(1,n+1) for y in range(x,n+1) for z in range(y,n+1) if x**2 + y**2 == z**2)
