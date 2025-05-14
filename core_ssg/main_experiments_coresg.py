import sys
import time
from experiments import runhdbscan

if __name__ == "__main__":

    print(sys.argv[1], sys.argv[2], sep=' ', end=' ', flush=True)

    start = time.time()

    runhdbscan.dbmsts(
        datafile=sys.argv[1], 
        kmax=int(sys.argv[2]), 
        delimiter=sys.argv[3], 
        method=sys.argv[4],
        efficient=True)
    
    end = time.time()
    print(str(end - start))
