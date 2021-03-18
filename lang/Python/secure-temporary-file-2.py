fd, path = tempfile.mkstemp()
try:
    # use the path or the file descriptor
    path
finally:
    os.close(fd)
