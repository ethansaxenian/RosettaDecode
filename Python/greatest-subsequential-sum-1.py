def maxsubseq(seq):
  return max((seq[begin:end] for begin in range(len(seq)+1)
                             for end in range(begin, len(seq)+1)),
             key=sum)
