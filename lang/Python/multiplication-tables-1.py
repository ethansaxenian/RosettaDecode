size = 12
width = len(str(size**2))
for row in range(-1,size+1):
	if row==0:
		print("─"*width + "┼"+"─"*((width+1)*size-1))
	else:
		print("".join("%*s%1s" % ((width,) + (("x","│")      if row==-1 and col==0
					              else (row,"│") if row>0   and col==0
					              else (col,"")  if row==-1
					              else ("","")   if row>col
					              else (row*col,"")))
			       for col in range(size+1)))

		
  x│  1   2   3   4   5   6   7   8   9  10  11  12
───┼───────────────────────────────────────────────
  1│  1   2   3   4   5   6   7   8   9  10  11  12
  2│      4   6   8  10  12  14  16  18  20  22  24
  3│          9  12  15  18  21  24  27  30  33  36
  4│             16  20  24  28  32  36  40  44  48
  5│                 25  30  35  40  45  50  55  60
  6│                     36  42  48  54  60  66  72
  7│                         49  56  63  70  77  84
  8│                             64  72  80  88  96
  9│                                 81  90  99 108
 10│                                    100 110 120
 11│                                        121 132
 12│                                            144

