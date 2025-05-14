from coressg.hdbscan import HDBSCAN

def dbmsts(datafile, kmin = 1, kmax = 16, delimiter=' ', method='core', efficient=True):
    
    h = HDBSCAN(datafile, min_pts=kmax, delimiter=delimiter)

    if method == 'core':
        h._hdbscan_knn(kmin=kmin, kmax=kmax)

    return None       
