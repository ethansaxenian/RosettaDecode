help('pprint.pprint')

from pprint import pprint

for tree in [(1, 2, 3, 4, 5, 6, 7, 8),
             (1, ((2, 3), (4, (5, ((6, 7), 8))))),
             ((((1, 2), 3), 4), 5, 6, 7, 8)]:
    print("\nTree %r can be pprint'd as:" % (tree,))
    pprint(tree, indent=1, width=1)

(1,
 2,
 3,
 4,
 5,
 6,
 7,
 8)

(1,
 ((2,
   3),
  (4,
   (5,
    ((6,
      7),
     8)))))

((((1,
    2),
   3),
  4),
 5,
 6,
 7,
 8)
