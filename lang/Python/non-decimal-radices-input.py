text = '100'
for base in range(2,21):
    print("String '%s' in base %i is  %i in base 10"
           % (text, base, int(text, base)))
