import subprocess
stat, out = subprocess.getstatusoutput('ls')
if not stat:
    print(out)
