import numpy as np

n = eval(input('Number of samples: '))
print(np.sum(np.random.rand(n)**2+np.random.rand(n)**2<1)/float(n)*4)
