def middle_three_digits(i):
	s = str(abs(i))
	length = len(s)
	assert length >= 3 and length % 2 == 1, "Need odd and >= 3 digits"
	mid = length // 2
	return s[mid-1:mid+2]

passing = [123, 12345, 1234567, 987654321, 10001, -10001, -123, -100, 100, -12345]
failing = [1, 2, -1, -10, 2002, -2002, 0]
for x in passing + failing:
	try:
		answer = middle_three_digits(x)
	except AssertionError as error:
		answer = error
	print("middle_three_digits(%s) returned: %r" % (x, answer))

	


