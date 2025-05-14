from coressg.coressg import SCORESG

def dbmsts(datafile, kmin = 1, kmax = 16, delimiter=' ', method='core', efficient=True, epsilon = 1):
    
    h = SCORESG(datafile, min_pts=kmax, delimiter=delimiter, epsilon = epsilon)

    if method == 'core':
        h._score_sg(kmin=kmin, kmax=kmax)

    return None       
