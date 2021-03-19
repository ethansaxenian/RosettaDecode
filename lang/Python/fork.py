import os

pid = os.fork()
if pid > 0:
    pass
# parent code
else:
    pass
 # child code
