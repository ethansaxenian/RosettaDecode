try:
    raw_input
except:
    raw_input = input

print((sum(map(int, input().split()))))
