myDict = {"hello": 13,
          "world": 31,
          "!": 71}

# iterating over key-value pairs:
for key, value in list(myDict.items()):
    print(("key = %s, value = %s" % (key, value)))

# iterating over keys:
for key in myDict:
    print(("key = %s" % key))
# (is a shortcut for:)
for key in list(myDict.keys()):
    print(("key = %s" % key))

# iterating over values:
for value in list(myDict.values()):
    print(("value = %s" % value))
