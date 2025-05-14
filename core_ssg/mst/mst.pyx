from __future__ import print_function

import pyximport
pyximport.install()

import numpy as np
cimport numpy as np

cimport cython

from disjoint_set import DisjointSet
from scipy.spatial import distance
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from libc.math cimport INFINITY
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

import heapq

include '../parameters.pxi'

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef _prim(
    DTYPE_t[:, :] data, 
    DTYPE_t[:] core_distances, 
    ITYPE_t self_edges):

    cdef ITYPE_t n, num_edges, num_edges_attached, current_point, nearest_point, neighbor
    cdef DTYPE_t nearest_distance, d 

    n = data.shape[0]

    num_edges = 2*n - 1 if self_edges else n - 1

    # keeps track of which points are attached to the tree.
    cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

    # arrays to keep track of the shortest connection to each point.
    cdef ITYPE_t[:] nearest_points = np.zeros(num_edges, dtype=ITYPE)
    cdef DTYPE_t[:] nearest_distances  = np.full(num_edges, np.inf, dtype=DTYPE)
    cdef DTYPE_t[:] distances_array 

    # keeps track of the number of edges added so far.
    num_edges_attached = 0

    # sets current point to the last point in the data.
    current_point = n - 1

    while (num_edges_attached < n - 1):

        # keeps track of the closest point to the tree.
        nearest_distance = INFINITY
        nearest_point = -1

        # marks current point as attached
        attached[current_point] = 1

        distances_array = distance.cdist([data[current_point]], data)[0]

        # loops over the dataset to find the next point to attach.
        for neighbor in xrange(n):    
            if attached[neighbor]: continue

            d = max(
                distances_array[neighbor],
                core_distances[current_point], 
                core_distances[neighbor])

            # updates the closese point to neigbor.
            if d < nearest_distances[neighbor]:
                nearest_distances[neighbor] = d
                nearest_points[neighbor] = current_point
            
            # updates the closest point to the tree. 
            if nearest_distances[neighbor] < nearest_distance:
                nearest_distance = nearest_distances[neighbor]
                nearest_point = neighbor

        # attached nearest_point to the tree.
        current_point = nearest_point
        
        # updates the number of edges added.
        num_edges_attached += 1

    # if self_edges:
        # nearest_points[n-1:] = np.arange(n)
        # nearest_distances[n-1:] = [ distance.euclidean(data[i], data[i]) for i in range(n)]
    
    return csr_matrix((nearest_distances, (nearest_points, np.arange(n-1))), shape=(n, n))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef prim_plus(
    DTYPE_t[:, :] data, 
    DTYPE_t[:] core_distances, 
    ITYPE_t[:] knn, 
    ITYPE_t self_edges):

    cdef ITYPE_t n, n_edges, num_edges_attached, current_point, nearest_point, neighbor, count_ties
    cdef DTYPE_t nearest_distance, d, core_current, dist, dummy

    n = data.shape[0]

    n_edges = 2*n - 1 if self_edges else n - 1

    # keeps track of which points are attached to the tree.
    cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

    # arrays to keep track of the shortest connection to each point.
    cdef ITYPE_t[:] nearest_points = np.zeros(n_edges, dtype=ITYPE)
    cdef DTYPE_t[:] nearest_distances  = np.full(n_edges, np.inf, dtype=DTYPE)

    # array to keep track of the distance from a point to the remaining points in the data.
    cdef DTYPE_t[:] distances_array

    # sparse matrix to store potential ties in the k-NN.
    # a_knn = dok_matrix((n, n), dtype=DTYPE)

    cdef list u = []
    cdef list v = []
    cdef list w = []

    # keeps track of the number of edges added so far.
    num_edges_attached = 0

    # sets current point to the last point in the data.
    current_point = n - 1

    while (num_edges_attached < n - 1):

        # keeps track of the closest point to the tree.
        nearest_distance = INFINITY
        nearest_point = -1

        # marks current point as attached
        attached[current_point] = 1

        distances_array = distance.cdist([data[current_point]], data)[0]

        # loops over the dataset to find the next point to attach.
        for neighbor in range(n):    

            if attached[neighbor]: continue

            dist = distances_array[neighbor]

            # includes ties in the k-NN
            if (dist == core_distances[current_point] and neighbor != knn[current_point]) or \
               (dist == core_distances[neighbor] and current_point != knn[neighbor]):
                u.append(current_point)
                v.append(neighbor)
                w.append(dist)

            d = max(
                dist,
                core_distances[current_point],
                core_distances[neighbor])

            # updates the closese point to neigbor.
            if d < nearest_distances[neighbor]:
                nearest_distances[neighbor] = d
                nearest_points[neighbor] = current_point
            
            # updates the closest point to the tree. 
            if nearest_distances[neighbor] < nearest_distance:
                nearest_distance = nearest_distances[neighbor]
                nearest_point = neighbor

        # attached nearest_point to the tree.
        current_point = nearest_point

        # updates the number of edges added.
        num_edges_attached += 1

    # if self_edges:
        # nearest_points[n-1:] = np.arange(n)
        # nearest_distances[n-1:] = [ distance.euclidean(data[i], data[i]) for i in range(n)]
    
    return csr_matrix((nearest_distances, (nearest_points, np.arange(n-1))), shape=(n, n)), \
           csr_matrix((w, (u, v)), shape=(n, n))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef _prim_inc(
    DTYPE_t[:, :] data,
    int[:] mst_indices,
    int[:] mst_indptr,
    DTYPE_t[:] mst_data,
    int[:] knng_indices,
    int[:] knng_indptr,
    DTYPE_t[:] knng_data,
    DTYPE_t[:] core_distances, 
    ITYPE_t min_pts, 
    ITYPE_t self_edges):

    cdef ITYPE_t n, n_edges, num_edges_attached, current_point, nearest_point, neighbor, i, n_current
    cdef DTYPE_t nearest_distance, d

    n = data.shape[0]

    n_edges = 2*n - 1 if self_edges else n - 1

    # keeps track of which points are attached to the tree.
    cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

    # arrays to keep track of the shortest connection to each point.
    cdef ITYPE_t[:] nearest_points = np.zeros(n_edges, dtype=ITYPE)
    cdef DTYPE_t[:] nearest_distances  = np.full(n_edges, np.inf, dtype=DTYPE)

    cdef DTYPE_t[:] distances_array 

    cdef FibonacciHeap heap
    cdef FibonacciNode* nodes = <FibonacciNode*> malloc(n * sizeof(FibonacciNode))

    for i in xrange(n):
        initialize_node(&nodes[i], i)

    cdef FibonacciNode *v
    cdef FibonacciNode *current_neighbor

    heap.min_node = NULL
    insert_node(&heap, &nodes[n-1])

    # keeps track of the number of edges added so far.
    num_edges_attached = 0

    # sets current point to the last point in the data.
    current_point = n - 1

    while (num_edges_attached < n - 1):

        v = remove_min(&heap)
        v.state = SCANNED

        # retrieves the closest point to the tree.
        current_point = v.index

        # loops over the MST.
        for i in xrange(mst_indptr[current_point], mst_indptr[current_point+1]):
            current_neighbor = &nodes[mst_indices[i]]

            if current_neighbor.state != SCANNED: 

                d = max(
                    mst_data[i],
                    core_distances[current_point], 
                    core_distances[current_neighbor.index])

                if current_neighbor.state == NOT_IN_HEAP:
                    current_neighbor.state = IN_HEAP
                    current_neighbor.val   = d + mst_data[i]
                    insert_node(&heap, current_neighbor)

                    nearest_distances[current_neighbor.index] = d
                    nearest_points[current_neighbor.index] = current_point

                elif d + mst_data[i] < current_neighbor.val:
                    decrease_val(&heap, current_neighbor, d + mst_data[i])

                    nearest_distances[current_neighbor.index] = d
                    nearest_points[current_neighbor.index] = current_point

        # loop over kNNG and removes edges that won't be needed anymore.
        for i in xrange(knng_indptr[current_point], knng_indptr[current_point+1]):
            current_neighbor = &nodes[knng_indices[i]]

            if current_neighbor.state != SCANNED: 
                
                d = max(
                    knng_data[i],
                    core_distances[current_point], 
                    core_distances[current_neighbor.index])

                if current_neighbor.state == NOT_IN_HEAP:
                    current_neighbor.state = IN_HEAP
                    current_neighbor.val   = d + knng_data[i]
                    insert_node(&heap, current_neighbor)

                    nearest_distances[current_neighbor.index] = d
                    nearest_points[current_neighbor.index] = current_point

                elif d + knng_data[i] < current_neighbor.val:
                    decrease_val(&heap, current_neighbor, d + knng_data[i])

                    nearest_distances[current_neighbor.index] = d
                    nearest_points[current_neighbor.index] = current_point

            # removes edges that won't be needed in the future
            if knng_data[i] > core_distances[current_point] and knng_data[i] > core_distances[current_neighbor.index]:
                knng_data[i] = 0

        # updates the number of edges added.
        num_edges_attached += 1

    # if self_edges:
        # nearest_points[n-1:] = np.arange(n)
        # nearest_distances[n-1:] = [ distance.euclidean(data[i], data[i]) for i in range(n)]
    
    free(nodes)

    return csr_matrix((nearest_distances, (nearest_points, np.arange(n-1))), shape=(n, n))



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef _prim_graph(
    int[:] indices,
    int[:] indptr,
    DTYPE_t[:] data,
    DTYPE_t[:] core_distances, 
    ITYPE_t self_edges):

    cdef ITYPE_t n, n_edges, num_edges_attached, current_point, nearest_point, neighbor, i, n_current
    cdef DTYPE_t nearest_distance, d

    n = core_distances.shape[0]

    n_edges = 2*n - 1 if self_edges else n - 1

    # keeps track of which points are attached to the tree.
    cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

    # arrays to keep track of the shortest connection to each point.
    cdef ITYPE_t[:] nearest_points = np.zeros(n_edges, dtype=ITYPE)
    cdef DTYPE_t[:] nearest_distances  = np.full(n_edges, np.inf, dtype=DTYPE)

    cdef DTYPE_t[:] distances_array 

    cdef FibonacciHeap heap
    cdef FibonacciNode* nodes = <FibonacciNode*> malloc(n * sizeof(FibonacciNode))

    for i in xrange(n):
        initialize_node(&nodes[i], i)

    cdef FibonacciNode *v
    cdef FibonacciNode *current_neighbor

    heap.min_node = NULL
    insert_node(&heap, &nodes[n-1])

    # keeps track of the number of edges added so far.
    num_edges_attached = 0

    # sets current point to the last point in the data.
    current_point = n - 1

    while (num_edges_attached < n - 1):

        v = remove_min(&heap)
        v.state = SCANNED

        # retrieves the closest point to the tree.
        current_point = v.index

        # loop over kNNG and removes edges that won't be needed anymore.
        for i in xrange(indptr[current_point], indptr[current_point+1]):

            current_neighbor = &nodes[indices[i]]

            if current_neighbor.state != SCANNED: 
                
                d = max(
                    data[i],
                    core_distances[current_point], 
                    core_distances[current_neighbor.index])

                if current_neighbor.state == NOT_IN_HEAP:
                    current_neighbor.state = IN_HEAP
                    current_neighbor.val   = d
                    insert_node(&heap, current_neighbor)

                    nearest_distances[current_neighbor.index] = d
                    nearest_points[current_neighbor.index] = current_point

                elif d < current_neighbor.val:
                    decrease_val(&heap, current_neighbor, d)

                    nearest_distances[current_neighbor.index] = d
                    nearest_points[current_neighbor.index] = current_point
        
        # updates the number of edges added.
        num_edges_attached += 1

    # if self_edges:
        # nearest_points[n-1:] = np.arange(n)
        # nearest_distances[n-1:] = [ distance.euclidean(data[i], data[i]) for i in range(n)]
    
    free(nodes)

    return csr_matrix((nearest_distances, (nearest_points, np.arange(n-1))), shape=(n, n))




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef prim_order(
    DTYPE_t[:] data,
    int[:] indices,
    int[:] indptr,
    int n):

    cdef ITYPE_t n_edges, num_points_attached, current_point, nearest_point, neighbor, i
    cdef DTYPE_t nearest_distance, d, weight

    n_edges = n - 1

    # keeps track of which points are attached to the tree.
    cdef ITYPE_t[:] attached = np.zeros(n, dtype=ITYPE)

    # arrays to keep track of the shortest connection to each point.
    cdef ITYPE_t[:] nearest_points = np.zeros(n_edges, dtype=ITYPE)
    cdef DTYPE_t[:] nearest_distances  = np.full(n_edges, np.inf, dtype=DTYPE)

    cdef DTYPE_t[:] distances_array 

    cdef ITYPE_t[:] order_p = np.zeros(n_edges, dtype=ITYPE)
    cdef DTYPE_t[:] order_w = np.zeros(n_edges, dtype=DTYPE)

    # keeps track of the number of edges added so far.
    num_points_attached = 0

    # sets current point to the last point in the data.
    current_point = n - 1

    pq = []
    
    heapq.heappush(pq, (0, current_point))

    while (num_points_attached < n - 1):

        # retrieves the closest point to the tree.
        weight, current_point = heapq.heappop(pq)

        order_p[num_points_attached] = current_point
        order_w[num_points_attached] = weight

        # attaches current_point and marks it as attached.
        attached[current_point] = 1

        # loops over the MST.
        for i in xrange(indptr[current_point], indptr[current_point+1]):
            neighbor = indices[i]

            if attached[neighbor]: continue

            d = data[i]

            if d < nearest_distances[neighbor]:
                nearest_distances[neighbor] = d
                nearest_points[neighbor] = current_point
                heapq.heappush(pq, (d, neighbor))
        
        # updates the number of edges added.
        num_points_attached += 1
    
    return order_p, order_w



#MST calculation with initial edges set by 1-knng
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef _split_mst(data, nn_distances, knn, mpts, self_edges):

    #auxiliary data structures
    #data size
    n = len(data)
    #disjoint set
    ds = DisjointSet()
    #creating matrix of mrd distances
    mrd = np.zeros((n,mpts))
    #matrix for MST edges
    mst_edges = np.zeros((n-1,2), dtype=int)
    #array of MST weights
    mst_weights = np.zeros(n-1)
    #edges count
    nedges = 0


    #init MST with closest neighbour
    for i in range(n): 
        #flag that indicates connection
        connected = False
        #calculating mrd for all points and building 1-NNG
        mrd[i,0] = nn_distances[i,mpts-1]
        for neighbor in range(1,mpts):
            mrd[i,neighbor] = max(nn_distances[i,neighbor],
            nn_distances[i, mpts-1],
            nn_distances[knn[i,neighbor], mpts-1])
            #building 1-NNG
            if (mrd[i,neighbor] == mrd[i,0] and not(connected)):
                if not(ds.connected(i,knn[i,neighbor])):
                    ds.union(i,knn[i,neighbor])
                    mst_edges[nedges] = (i,knn[i,neighbor])
                    mst_weights[nedges] = mrd[i,neighbor]
                    nedges += 1
                    connected = True
        #insert singleton point in sets if not connected			
        if not(connected): 
            ds.find(i)		
                        
    #Updating sets info
    sets = list(ds.itersets()) 
    nsets = len(sets)
    inter_mrd = np.matrix(np.ones((nsets,nsets)) * np.inf)
    #Updating self-edges
    for i in range(nsets):
        inter_mrd[i,i] = 0	
    inter_edges = np.zeros((nsets,nsets,2), dtype=int)

    #Inter-sets distances calculation and selection
    for i in range(nsets):
        for j in range(i+1,nsets):
            #conveting sets to lists
            setlist1 = list(sets[i])
            setlist2 = list(sets[j])
            D = distance.cdist(data[setlist1],data[setlist2],'euclidean')
            #finding lowest mrd inter-set 
            for l in range(len(setlist1)):
                for m in range(len(setlist2)):
                    aux = max(D[l,m], mrd[setlist1[l],0], mrd[setlist2[m],0])
                    if inter_mrd[i,j] > aux:
                        inter_mrd[i,j] = aux
                        inter_mrd[j,i] = aux
                        inter_edges[i,j] = (setlist1[l],setlist2[m])
                        inter_edges[j,i] = inter_edges[i,j]
                                                            
    #building MST of the initial sets
    inter_mst = minimum_spanning_tree(inter_mrd)

    #converting interset MST into global MST
    (ind1,ind2) = inter_mst.nonzero()
    for i in range(len(ind1)):
        mst_edges[nedges] = inter_edges[ind1[i],ind2[i]]
        mst_weights[nedges] = inter_mrd[ind1[i],ind2[i]]
        nedges += 1
        
    #temporary debug
    if nedges < (n-1):
        print("Faltou aresta " + str(nedges))
        
    #building and returning MST matrix
    return csr_matrix((mst_weights, (mst_edges[:,0], mst_edges[:,1])), shape=(n, n))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef prim(DTYPE_t[:, :] data, DTYPE_t[:] core_distances, np.int64_t self_edges):
    return _prim(data, core_distances, self_edges)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef prim_inc(DTYPE_t[:, :] data, int[:] mst_indices, int[:] mst_indptr, DTYPE_t[:] mst_data, int[:] knng_indices, int[:] knng_indptr, DTYPE_t[:] knng_data, DTYPE_t[:] core_distances, ITYPE_t min_pts, ITYPE_t self_edges):
    return _prim_inc(data, mst_indices, mst_indptr, mst_data, knng_indices, knng_indptr, knng_data, core_distances, min_pts, self_edges)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef prim_graph(int[:] indices, int[:] indptr, DTYPE_t[:] data, DTYPE_t[:] core_distances, ITYPE_t self_edges):
    return _prim_graph(indices, indptr, data, core_distances, self_edges)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef split_mst(data, nn_distances, knn, mpts, self_edges):
    return _split_mst(data, nn_distances, knn, mpts, self_edges)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef DTYPE_t euclidean_local(DTYPE_t[:] v1, DTYPE_t[:] v2):
    cdef ITYPE_t i, m
    cdef DTYPE_t d = 0.0
    m = v1.shape[0]

    for i in xrange(m):
        d += (v1[i] - v2[i])**2

    return sqrt(d)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef DTYPE_t euclidean_export(DTYPE_t[:] v1, DTYPE_t[:] v2):
    cdef ITYPE_t i, m
    cdef DTYPE_t d = 0.0
    m = v1.shape[0]

    for i in xrange(m):
        d += (v1[i] - v2[i])**2

    return sqrt(d)


######################################################################
# FibonacciNode structure
#  This structure and the operations on it are the nodes of the
#  Fibonacci heap.
#
cdef enum FibonacciState:
    SCANNED
    NOT_IN_HEAP
    IN_HEAP


cdef struct FibonacciNode:
    unsigned int index
    unsigned int rank
    unsigned int source
    FibonacciState state
    DTYPE_t val
    FibonacciNode* parent
    FibonacciNode* left_sibling
    FibonacciNode* right_sibling
    FibonacciNode* children


cdef void initialize_node(FibonacciNode* node,
                          unsigned int index,
                          DTYPE_t val=0):
    # Assumptions: - node is a valid pointer
    #              - node is not currently part of a heap
    node.index = index
    node.source = -9999
    node.val = val
    node.rank = 0
    node.state = NOT_IN_HEAP

    node.parent = NULL
    node.left_sibling = NULL
    node.right_sibling = NULL
    node.children = NULL


cdef FibonacciNode* rightmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.right_sibling):
        temp = temp.right_sibling
    return temp


cdef FibonacciNode* leftmost_sibling(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    cdef FibonacciNode* temp = node
    while(temp.left_sibling):
        temp = temp.left_sibling
    return temp


cdef void add_child(FibonacciNode* node, FibonacciNode* new_child):
    # Assumptions: - node is a valid pointer
    #              - new_child is a valid pointer
    #              - new_child is not the sibling or child of another node
    new_child.parent = node

    if node.children:
        add_sibling(node.children, new_child)
    else:

        node.children = new_child
        new_child.right_sibling = NULL
        new_child.left_sibling = NULL
        node.rank = 1


cdef void add_sibling(FibonacciNode* node, FibonacciNode* new_sibling):
    # Assumptions: - node is a valid pointer
    #              - new_sibling is a valid pointer
    #              - new_sibling is not the child or sibling of another node
    cdef FibonacciNode* temp = rightmost_sibling(node)
    temp.right_sibling = new_sibling
    new_sibling.left_sibling = temp
    new_sibling.right_sibling = NULL
    new_sibling.parent = node.parent
    if new_sibling.parent:
        new_sibling.parent.rank += 1


cdef void remove(FibonacciNode* node):
    # Assumptions: - node is a valid pointer
    if node.parent:
        node.parent.rank -= 1
        if node.left_sibling:
            node.parent.children = node.left_sibling
        elif node.right_sibling:
            node.parent.children = node.right_sibling
        else:
            node.parent.children = NULL

    if node.left_sibling:
        node.left_sibling.right_sibling = node.right_sibling
    if node.right_sibling:
        node.right_sibling.left_sibling = node.left_sibling

    node.left_sibling = NULL
    node.right_sibling = NULL
    node.parent = NULL


######################################################################
# FibonacciHeap structure
#  This structure and operations on it use the FibonacciNode
#  routines to implement a Fibonacci heap

ctypedef FibonacciNode* pFibonacciNode


cdef struct FibonacciHeap:
    FibonacciNode* min_node
    pFibonacciNode[100] roots_by_rank  # maximum number of nodes is ~2^100.


cdef void insert_node(FibonacciHeap* heap,
                      FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    if heap.min_node:
        add_sibling(heap.min_node, node)
        if node.val < heap.min_node.val:
            heap.min_node = node
    else:
        heap.min_node = node


cdef void decrease_val(FibonacciHeap* heap,
                       FibonacciNode* node,
                       DTYPE_t newval):
    # Assumptions: - heap is a valid pointer
    #              - newval <= node.val
    #              - node is a valid pointer
    #              - node is not the child or sibling of another node
    #              - node is in the heap
    node.val = newval
    if node.parent and (node.parent.val >= newval):
        remove(node)
        insert_node(heap, node)
    elif heap.min_node.val > node.val:
        heap.min_node = node


cdef void link(FibonacciHeap* heap, FibonacciNode* node):
    # Assumptions: - heap is a valid pointer
    #              - node is a valid pointer
    #              - node is already within heap

    cdef FibonacciNode *linknode
    cdef FibonacciNode *parent
    cdef FibonacciNode *child

    if heap.roots_by_rank[node.rank] == NULL:
        heap.roots_by_rank[node.rank] = node
    else:
        linknode = heap.roots_by_rank[node.rank]
        heap.roots_by_rank[node.rank] = NULL

        if node.val < linknode.val or node == heap.min_node:
            remove(linknode)
            add_child(node, linknode)
            link(heap, node)
        else:
            remove(node)
            add_child(linknode, node)
            link(heap, linknode)


cdef FibonacciNode* remove_min(FibonacciHeap* heap):
    # Assumptions: - heap is a valid pointer
    #              - heap.min_node is a valid pointer
    cdef:
        FibonacciNode *temp
        FibonacciNode *temp_right
        FibonacciNode *out
        unsigned int i

    # make all min_node children into root nodes
    if heap.min_node.children:
        temp = leftmost_sibling(heap.min_node.children)
        temp_right = NULL

        while temp:
            temp_right = temp.right_sibling
            remove(temp)
            add_sibling(heap.min_node, temp)
            temp = temp_right

        heap.min_node.children = NULL

    # choose a root node other than min_node
    temp = leftmost_sibling(heap.min_node)
    if temp == heap.min_node:
        if heap.min_node.right_sibling:
            temp = heap.min_node.right_sibling
        else:
            out = heap.min_node
            heap.min_node = NULL
            return out

    # remove min_node, and point heap to the new min
    out = heap.min_node
    remove(heap.min_node)
    heap.min_node = temp

    # re-link the heap
    for i in range(100):
        heap.roots_by_rank[i] = NULL

    while temp:
        if temp.val < heap.min_node.val:
            heap.min_node = temp
        temp_right = temp.right_sibling
        link(heap, temp)
        temp = temp_right

    return out