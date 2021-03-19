class MyClassName:
	__private = 123
	non_private = __private * 2

	
mine = MyClassName()
mine.non_private
246
mine.__private
mine._MyClassName__private
123

