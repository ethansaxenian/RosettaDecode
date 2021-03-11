
from multiprocessing import Pool

def main():
    p = Pool()
    p.map(print, 'Enjoy Rosetta Code'.split())

if __name__=="__main__":
    main()
