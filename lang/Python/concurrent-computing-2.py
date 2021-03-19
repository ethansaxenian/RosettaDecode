
from concurrent import futures
with futures.ProcessPoolExecutor() as executor:
   _ = list(executor.map(print, 'Enjoy Rosetta Code'.split()))

