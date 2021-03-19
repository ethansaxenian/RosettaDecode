for n in (254, 255, 256, 257, -2+(1<<16), -1+(1<<16), 1<<16, 1+(1<<16), 0x200000, 0x1fffff ):
    print('int: %7i bin: %26s vlq: %35s vlq->int: %7i' % (n, tobits(n,_pad=True), tovlq(n), toint(tovlq(n))))




