for i in range(1000): world[choice(possiblepoints)] += 1

for x in range(-15,16):
	print(''.join(str(min([9, world[(x,y)]])) if world[(x,y)] else ' '
			  for y in range(-15,16)))

