#Python 2.X
import urllib.request, urllib.parse, urllib.error
print(urllib.parse.unquote("http%3A%2F%2Ffoo%20bar%2F"))
#Python 3.5+
from urllib.parse import unquote
print((unquote('http%3A%2F%2Ffoo%20bar%2F')))
