lettercounts = countletters(sourcedata)
for letter,count in lettercounts.items():
    print("%s=%s" % (letter, count), end=' ')
