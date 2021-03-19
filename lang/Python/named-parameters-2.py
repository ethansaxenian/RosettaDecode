from __future__ import print_function

def show_args(defparam1, defparam2 = 'default value', *posparam, **keyparam):
  "Straight-forward function to show its arguments"
  print("  Default Parameters:")
  print("    defparam1 value is:", defparam1)
  print("    defparam2 value is:", defparam2)

  print("  Positional Arguments:")
  if posparam:
    n = 0
    for p in posparam:
      print("    positional argument:", n, "is:", p)
      n += 1
  else:
    print("    <None>")

  print("  Keyword Arguments (by sorted key name):")
  if keyparam:
    for k,v in sorted(keyparam.items()):
      print("    keyword argument:", k, "is:", v)
  else:
    print("    <None>")


show_args('POSITIONAL', 'ARGUMENTS')

show_args(defparam2='ARGUMENT', defparam1='KEYWORD')

show_args( *('SEQUENCE', 'ARGUMENTS') )

show_args( **{'defparam2':'ARGUMENTS', 'defparam1':'MAPPING'} )

show_args('ONLY DEFINE defparam1 ARGUMENT')

show_args('POSITIONAL', 'ARGUMENTS',
              'EXTRA', 'POSITIONAL', 'ARGUMENTS')

show_args('POSITIONAL', 'ARGUMENTS',
              kwa1='EXTRA', kwa2='KEYWORD', kwa3='ARGUMENTS')

show_args('POSITIONAL',
              'ARGUMENTS', 'EXTRA', 'POSITIONAL', 'ARGUMENTS',
              kwa1='EXTRA', kwa2='KEYWORD', kwa3='ARGUMENTS')

# But note:
show_args('POSITIONAL', 'ARGUMENTS',
              kwa1='EXTRA', kwa2='KEYWORD', kwa3='ARGUMENTS')

