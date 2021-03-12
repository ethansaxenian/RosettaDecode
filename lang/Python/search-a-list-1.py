haystack=["Zig","Zag","Wally","Ronald","Bush","Krusty","Charlie","Bush","Bozo"]

for needle in ("Washington","Bush"):
  try:
    print(haystack.index(needle), needle)
  except ValueError as value_error:
    print(needle,"is not in haystack")
