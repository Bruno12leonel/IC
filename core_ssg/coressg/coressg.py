import sys
import os
import time
import typing

import numpy as np

from scipy.sparse import csr_matrix, triu
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import MinMaxScaler


from mst.mst import prim
from mst.mst import prim_plus
from mst.mst import prim_inc
from mst.mst import prim_graph
from mst.mst import prim_order

from databubble.databubble import DataBubble


class SCORESG:

    def __init__(
        self, 
        datafile, 
        min_pts = 16, 
        delimiter=',', 
        distance='euclidean', 
        skip=2,
        epsilon = 1):

        sys.setrecursionlimit(10**6)

        # load data.
        try:
            self.data = np.unique(np.genfromtxt(datafile, delimiter=',', skip_header=1), axis=0)
        except:
            print("Error reading the data file, please verify that the file exists.")        

        # finds the number of points in the data.
        self.n = len(self.data)
        self.epsilon = epsilon
        self.mu = 0
        self.p_micro_clusters: typing.Dict[int, "DataBubble"] = {}
        
        # value of min_pts must be at most the number of points in the data.
        self.min_pts = min(self.n, min_pts)        
        
        # determines the distance function to be used for clustering.
        self.distance = distance

        # determines the interval between min_pts values in the range.
        self.skip = skip

        name = os.path.splitext(datafile)[0]

        ############################################################################
        ##################         Summarization                  ##################              
        ############################################################################

        scaler = MinMaxScaler()

        scaler.fit(self.data)

        self.data = scaler.transform(self.data)



        visited = np.zeros(len(self.data), dtype=int)
        db_reps = []
        
        start = time.time()
        nbrs = NearestNeighbors( n_neighbors = self.min_pts).fit(self.data)
        
        distances, knn = nbrs.kneighbors(self.data)
        
        i = 0
        
        
        for item in distances:

            if not visited[i]:

                ids_dbs = []

                visited[i] = 1

                j = 0

                for neighbour_dist in item:

                    if not visited[knn[i][j]]:

                        if neighbour_dist < self.epsilon:
                            ids_dbs.append(knn[i][j]) 

                    j += 1

                if len(ids_dbs) > self.mu:
                    db = DataBubble(
                        x = dict(zip([dim for dim in range(len(self.data[i]))], self.data[i])),
                        timestamp = 0,
                        decaying_factor = 0,
                    )

                    for neighbour_id in ids_dbs:

                        if not visited[neighbour_id]:
                            visited[neighbour_id] = 1

                            db.insert(dict(zip([dim for dim in range(len(self.data[neighbour_id]))], self.data[neighbour_id])))

                    self.p_micro_clusters.update({len(self.p_micro_clusters): db})
                    db_reps.append(list(db.getRep().values()))

                else:
                    visited[i] = 0

            i+=1
            
                
        end = time.time()
        print("tempo para gerar os DBs",end - start, end=' ')
        
        print("\nnum db: ", len(self.p_micro_clusters))
        self.data = np.array(db_reps)
        
        start = time.time()
        nbrs = NearestNeighbors( n_neighbors = (self.min_pts//2) + 1).fit(db_reps)
        
        distances, knn = nbrs.kneighbors(db_reps)
        
        self.core_distances = np.zeros((len(self.p_micro_clusters),self.min_pts) , dtype=float)
        self.knn            = np.zeros((len(self.p_micro_clusters),self.min_pts), dtype = int)
        
                
        i = 0
        
        for item in distances:           

            j = 0
            
            db_current = self.p_micro_clusters[i]
            
            count_points = db_current.getN()

            for neighbour_dist in item:                
                
                if j == 0:
                    for x in range(db_current.getN()):
                        self.core_distances[i][x] = db_current.getNnDist(x + 1)
                        self.knn[i][x] = i
                    
                else:
                    id_db = knn[i][j]
                    
                    db_neighbour = self.p_micro_clusters[id_db]
                    
                    for x in range(db_neighbour.getN()):
                        if (count_points + x) == self.min_pts:
                            break
                        
                        self.core_distances[i][count_points + x] = self.ditanceDataBubbles(db_current, db_neighbour, neighbour_dist) + db.getNnDist(x + 1)
                        self.knn[i][count_points + x] = id_db
                        
                        
                    count_points += db_neighbour.getN()
                
                    if (count_points) >= self.min_pts:
                        break
                
                j += 1
            
            i+=1
        
        end = time.time()
        print("tempo para gerar o coreDistance",end - start, end=' ')

        self.n = len(self.core_distances)
        print(self.core_distances.shape)
        self.knng = self._knng(self.min_pts)


    def ditanceDataBubbles(self, db_current, db_neighbour, distance):
        x1 = distance - (db_current.getExtent() + db_neighbour.getExtent())
        x2 = db_current.getNnDist(1)
        x3 = db_neighbour.getNnDist(1)
        
        if x1 >= 0:
            return x1 + x2 + x3
        
        return max(x2, x3)
    
        ############################################################################
        ##################                                        ##################              
        ############################################################################

    
    def _score_sg(self, kmin = 1, kmax = 16):
        
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
        print("\ntempo da mst: ", end - start, end=' ')
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
        print("\ntempo no loop do coresg: ",end - start, end=' ')
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
