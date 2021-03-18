class Immut(object):
	def __setattr__(self, *args):
		raise TypeError(
			"'Immut' object does not support item assignment")
	
        __delattr__ = __setattr__
	
        def __repr__(self):
		return str(self.value)
	
        def __init__(self, value):
                # assign to the un-assignable the hard way.
		super(Immut, self).__setattr__("value", value)

im = Immut(123)
im
123
im.value = 124


