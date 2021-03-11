print('\n'.join(['Door %s is %s' % (i, ('closed', 'open')[(i ** 0.5).is_integer()]) for i in range(1, 101)]))
