import sys
import time
from experiments import runscoressg

if __name__ == "__main__":    

    start = time.time()

    runscoressg.dbmsts(
        datafile=sys.argv[1], 
        kmax=int(sys.argv[2]), 
        delimiter=sys.argv[3], 
        method=sys.argv[4],
        efficient=True,
        epsilon = sys.argv[5])
    
    end = time.time()
    print(str(end - start))
