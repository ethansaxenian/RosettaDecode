import bisect
_cin  = [ 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101]
_cout = [10, 18, 26, 32, 38, 44, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 100]
def centsrounder(centsin):
	return _cout[ bisect.bisect_right(_cin, centsin) ]
