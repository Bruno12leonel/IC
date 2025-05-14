import sys
import os
import time

import numpy as np

from scipy.sparse import csr_matrix, save_npz, load_npz, triu

from mst.mst import prim
from mst.mst import prim_plus
from mst.mst import prim_inc
from mst.mst import prim_graph
from mst.mst import prim_order

class HDBSCAN:

    def __init__(
        self, 
        datafile, 
        min_pts = 16, 
        delimiter=',', 
        distance='euclidean', 
        skip=2):

        sys.setrecursionlimit(10**6)

        # load data.
        try:
            self.data = np.unique(np.genfromtxt(datafile, delimiter=','), axis=0)
        except:
            print("Error reading the data file, please verify that the file exists.")        

        # finds the number of points in the data.
        self.n = len(self.data)
        
        # value of min_pts must be at most the number of points in the data.
        self.min_pts = min(self.n, min_pts)        
        
        # determines the distance function to be used for clustering.
        self.distance = distance

        # determines the interval between min_pts values in the range.
        self.skip = skip

        name = os.path.splitext(datafile)[0]

        try:
            self.core_distances = np.load(name + "-" + str(min_pts) + "-cd.npy")
            self.knn = np.load(name + "-" + str(min_pts) + "-knn.npy")
        except:
            from sklearn.neighbors import NearestNeighbors
            from mst.mst import euclidean_export

            if distance == 'precomputed':
                nbrs = NearestNeighbors(n_neighbors=min_pts, metric='precomputed').fit(self.data)
            else:
                nbrs = NearestNeighbors(n_neighbors=min_pts).fit(self.data)

            # computes the core-distances and knn information.
            self.core_distances, self.knn = nbrs.kneighbors(self.data)

            if distance != 'precomputed':
                # fix core-distances to use the same precision as the local function.
                for i in range(self.n):
                    for j in range(1, self.min_pts):
                        self.core_distances[i, j] = euclidean_export(self.data[i], self.data[self.knn[i, j]])

            # saving the computed core-distances, knn and knng on files.
            np.save(name + "-" + str(self.min_pts) + "-cd" , self.core_distances)
            np.save(name + "-" + str(self.min_pts) + "-knn", self.knn.astype(int))

        try:
            self.knng = load_npz(datafile + "-" + str(self.min_pts) + ".npz")
        except:
            # computes the kNNG
            self.knng = self._knng(self.min_pts)
            save_npz(datafile + "-" + str(self.min_pts) + ".npz", self.knng)

    
    def _hdbscan_knn(self, kmin = 1, kmax = 16):
        
        # -----------------------------------
        start = time.time()
        
        # computes the MST w.r.t kmax and returns augmented kmax-NN information.
        mst, a_knn = prim_plus(
            self.data, 
            np.ascontiguousarray(self.core_distances[:, kmax-1]), 
            np.ascontiguousarray(self.knn[:, kmax-1]),
            False)

        # makes the mst an upper triangular matrix.
        mst = triu(mst.maximum(mst.T), format='csr')

        # augments the knng with the ties.
        self.knng = self.knng.maximum(a_knn.maximum(a_knn.T))
        
        # computes the CORE-SG graph w.r.t. the underlying distance. 
        nnsg = self._nnsg(mst, triu(self.knng))

        # eliminates zeroes from the matrix that might have remained from the operations.
        nnsg.eliminate_zeros()
        
        nnsg = nnsg.maximum(nnsg.T)

        end = time.time()
        print(end - start, end=' ')
        # -----------------------------------

        # -----------------------------------
        start = time.time()

        # loop over the values of mpts in the input range [kmin, kmax].
        for i in range(kmin, kmax, self.skip): 

            # compute mst for mpts = i
            mst = prim_graph(
                nnsg.indices,
                nnsg.indptr,
                nnsg.data,
                np.ascontiguousarray(self.core_distances[:, i-1]),
                False)

        end = time.time()
        print(end - start, end=' ')
        # -----------------------------------
        #print(int(nnsg.count_nonzero()/2), end=' ')

    def _knng(self, min_pts):
        n_neighbors = min_pts - 1
        n_nonzero = self.n * n_neighbors

        knng = csr_matrix(
            (self.core_distances[:, 1:].ravel(), 
            self.knn[:, 1:].ravel(), 
            np.arange(0, n_nonzero + 1, n_neighbors)), 
            shape=(self.n, self.n))

        return knng.maximum(knng.T)


    def _nnsg(self, mst, knng, format='csr'):

        for current_point in range(mst.shape[0]):
            for i in range(mst.indptr[current_point], mst.indptr[current_point+1]):
                neighbor = mst.indices[i]
                if mst.data[i] == self.core_distances[current_point, self.min_pts-1] or \
                   mst.data[i] == self.core_distances[neighbor, self.min_pts-1]:
                   mst.data[i] = 0

        return mst.maximum(knng)


    def _construct_hierarchy(self, mst):
        
        dendrogram = {}

        # get order of edges.
        nodes, reachability = self._get_reachability(mst)

        # current nodes of a point in the dendrogram.
        current_nodes = np.arange(0, self.n)

        # index of points in order of reachability values.
        ordered_indices = np.argsort(reachability)

        # initialize dendrogram
        for i, r in zip(nodes, reachability):
            dendrogram[i] = {'id': i, 'height':r, 'left': None, 'right': None}

        new_node = self.n

        for i in range(1, self.n - 1):
            p = nodes[ordered_indices[i]]
            q = nodes[ordered_indices[i] - 1]

            dendrogram[new_node] = {'id': new_node, 
                                    'height':reachability[ordered_indices[i]], 
                                    'left': current_nodes[q],
                                    'right': current_nodes[p]}
            
            current_nodes[p] = new_node
            current_nodes[q] = new_node

            new_node += 1

        return dendrogram


    def _simplified_hierarchy(self, mst):
        
        mst = mst.maximum(mst.T)

        nodes, reachability = prim_order(mst.data, mst.indices, mst.indptr, self.n)

        return None


    def _get_reachability(self, mst):

        mst = mst.maximum(mst.T)

        nodes, reachability = prim_order(mst.data, mst.indices, mst.indptr, self.n)

        return nodes, reachability


    def _get_nodes(self, reachability, start, end):

        if end - start < 2:
            return None

        split = start + 1 + np.argmax(reachability[start+1:end])
        # print("-----------------------------------")
        # print(reachability[start+1:end])
        # print(start+1, end, split)

        d = {}

        d['level'] = reachability[split]
        d['start'] = start
        d['end']   = end
        
        d['children'] = [self._get_nodes(reachability, start, split),
        self._get_nodes(reachability, split + 1, end)]

        return d
