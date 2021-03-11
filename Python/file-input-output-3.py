import sys
try:
    infile = open('input.txt', 'r')
except IOError:
    print("Unable to open input.txt for input", file=sys.stderr)
    sys.exit(1)
try:
    outfile = open('output.txt', 'w')
except IOError:
    print("Unable to open output.txt for output", file=sys.stderr)
    sys.exit(1)
try:  # for finally
    try: # for I/O
        for line in infile:
            outfile.write(line)
    except IOError as e:
        print("Some I/O Error occurred (reading from input.txt or writing to output.txt)", file=sys.stderr)
finally:
    infile.close()
    outfile.close()
