import pickle as pickle

source = {'a': [1, 2.0, 3, 4+6j],
         'b': ('string', 'Unicode string'),
         'c': None}

target = pickle.loads(pickle.dumps(source))
