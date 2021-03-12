import hashlib
print(hashlib.new("md4",input().encode('utf-16le')).hexdigest().upper())
