from itertools import chain
def dutch_flag_sort2(items, order=colours_in_order):
    'return summed filter of items using the given order'
    return list(chain.from_iterable([c for c in items if c==colour]
                                    for colour in order))
