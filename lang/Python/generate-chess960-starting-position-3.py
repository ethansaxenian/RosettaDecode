from random import choice

def random960():
    start = ['R', 'K', 'R']         # Subsequent order unchanged by insertions.
    #
    for piece in ['Q', 'N', 'N']:
        start.insert(choice(list(range(len(start)+1))), piece)
    #
    bishpos = choice(list(range(len(start)+1)))
    start.insert(bishpos, 'B')
    start.insert(choice(list(range(bishpos + 1, len(start) + 1, 2))), 'B')
    return start
    return ''.join(start).upper()

print((random960()))
