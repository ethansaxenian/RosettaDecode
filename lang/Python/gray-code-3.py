for i in range(16):
	print('int:%2i -> bin:%12r -> gray:%12r -> bin:%12r -> int:%2i' %
	      ( i,
	        int2bin(i),
	        bin2gray(int2bin(i)),
	        gray2bin(bin2gray(int2bin(i))),
	        bin2int(gray2bin(bin2gray(int2bin(i))))
	      ))


