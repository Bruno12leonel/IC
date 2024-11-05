import river
import copy
import math
import typing
import time
import os
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
import sys

from river import stream
from river import base, utils
from abc import ABCMeta
from collections import defaultdict, deque
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

class SuperVertex():
    s_idCounterSuperVertex = 0

    def __init__(self, c):
        self.m_id        = SuperVertex.s_idCounterSuperVertex
        SuperVertex.s_idCounterSuperVertex += 1
        self.m_component = c
        self.m_vertices  = set(self.m_component.getVertices())
        self.m_visited   = False


    def getVertices(self):
        return self.m_vertices

    def visited(self) -> bool:
        return self.m_visited

    def setVisited(self):
        self.m_visited = True

    def resetVisited(self):
        self.m_visited = False

    def getID(self):
        return self.m_id

    def compareID(self, other: 'SuperVertex'):
        return self.m_id == other.m_id

    def isSuperVertex(self):
        return True

    def containsVertex(self, v):
        return v in self.m_vertices

    def getComponent(self):
        return self.m_component
    
class Vertex():
    
    s_idCounter = 0

    def __init__(self, mc, timestamp, id=None):
        self.m_id          = id if id is not None else Vertex.s_idCounter
        Vertex.s_idCounter += 1
        self.m_mc          = mc
        self.timestamp     = timestamp
        
        if self.m_mc is not None:
            self.m_mc.setVertexRepresentative(self)
            
        self.m_visited            = False
        self.m_coreDistanceObject = None
        self.m_lrd                = -1
        self.m_coreDist           = 0
        
    def getMicroCluster(self):
        return self.m_mc

    def getCoreDistance(self):
        if self.m_coreDist == 0:
            return -1
        return self.m_coreDist

    def setCoreDist(self, coreDistValue):
        self.m_coreDist = coreDistValue

    def setCoreDistance(self, coreDistObj):
        self.m_coreDistanceObject = coreDistObj

    def String(self):
        return f"({self.m_mc.getCenter()})"

    def getGraphVizVertexString(self):
        return f"vertex{self.m_id}"

    def getGraphVizString(self):
        return f"{self.getGraphVizVertexString()} [label='{self}';cdist={self.getCoreDistance()}]"

    def getDistanceToVertex(self, other):
        return self.m_mc.getCenterDistance(other.getMicroCluster())
    '''
    def getDistanceRep(self, vertex):
        x1 = self.distance(self.m_mc.getCenter(), vertex.getMicroCluster().getCenter())
        
        return x1
    '''
    def getDistance(self, vertex):
        if self.m_mc.getStaticCenter() is None or vertex.getMicroCluster().getStaticCenter() is None:
            return self.getDistanceToVertex(vertex)
        
        return self.distance(self.m_mc.getCenter(self.timestamp), vertex.getMicroCluster().getCenter(self.timestamp))

    def distance(self, v1, v2):
        distance = 0
        for i in range(len(v1)):
            d = v1[i] - v2[i]
            distance += d * d
        return math.sqrt(distance)

    def setCoreDistChanged(self):
        self.m_changeCoreDist = True

    def resetCoreDistChanged(self):
        self.m_changeCoreDist = False

    def hasCoreDistChanged(self):
        return self.m_changeCoreDist

    def visited(self):
        return self.m_visited

    def setVisited(self):
        self.m_visited = True

    def resetVisited(self):
        self.m_visited = False;

    def getID(self):
        return self.m_id

    def compareID(self,  other: "Vertex"):
        return self.m_id == other.m_id

class MicroCluster(metaclass=ABCMeta):

    s_idCounter = 0
    
    def __init__(self, x, timestamp, decaying_factor):

        self.x = x

        self.db_id           = MicroCluster.s_idCounter        
        self.last_edit_time  = timestamp
        self.creation_time   = timestamp
        self.decaying_factor = decaying_factor

        self.N              = 1
        self.linear_sum     = x
        self.squared_sum    = {i: (x_val * x_val) for i, x_val in x.items()}        
        self.m_staticCenter = [];

    def getID(self):
        return self.db_id

    def setID(self, id):
        self.db_id = id
    
    def calc_cf1(self, fading_function):
        cf1 = []        
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            cf1.append(fading_function * val_ls)
        return cf1
    
    def calc_cf2(self, fading_function):
        cf2 = []        
        for key in self.squared_sum.keys():
            val_ss = self.squared_sum[key]
            cf2.append(fading_function * val_ss)
        return cf2

    def calc_weight(self):
        return self._weight()
    
    def getN(self):
        return self.N

    def getWeight(self, timestamp):
        return self.N * self.fading_function(timestamp - self.last_edit_time)
        
    def getCenter(self, timestamp):
        ff = self.fading_function(timestamp - self.last_edit_time)
        weight = self.getWeight(timestamp)
        center = {key: (ff * val) / weight for key, val in self.linear_sum.items()}
        
        return center
    '''
    def getRadius(self, timestamp):        
        x1  = 0
        x2  = 0
        res = 0
        
        ff     = self.fading_function(timestamp - self.last_edit_time)
        weight = self.getWeight(timestamp)
        
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            val_ss = self.squared_sum[key]
            
            x1  = 2 * (val_ss * ff) * weight
            x2  = 2 * (val_ls * ff)**2
            tmp = (x1 - x2)
                        
            if tmp <= 0.0:
                tmp = 1/10 * 1/10
            
            diff = (tmp / (weight * (weight - 1))) if (weight * (weight - 1)) > 0.0 else 0.1
            
            res += math.sqrt(diff) if diff > 0 else 0

        return (res / len(self.linear_sum)) * 1.5  #redius factor
        #return res
    '''
    def getRadius(self, timestamp):        
        ff  = self.fading_function(timestamp - self.last_edit_time)
        w   = self.getWeight(timestamp)        
        cf1 = self.calc_cf1(ff)
        cf2 = self.calc_cf2(ff)        
        res = 0      
        
        for i in range(len(self.linear_sum)):
            x1 = cf2[i] / w
            x2 = math.pow(cf1[i]/w , 2)
            
            tmp = x1 - x2
            
            res += math.sqrt(tmp) if tmp > 0 else (1/10 * 1/10)
            
        #1.8            
        return (res / len(cf1)) * 1.8
        
    def add(self, x):        
        self.N += 1
        
        for key, val in x.items():
            self.linear_sum[key]  += val
            self.squared_sum[key] += val * val
            
    def insert(self, x, timestamp):
        if(self.last_edit_time != timestamp):
            self.fade(timestamp)
        
        self.last_edit_time = timestamp
        
        self.add(x)
    
    def fade(self, timestamp):
        ff = self.fading_function(timestamp - self.last_edit_time)
        
        self.N *= ff
        
        for key, val in x.items():            
            self.linear_sum[key]  *= ff
            self.squared_sum[key] *= ff
        
    def merge(self, cluster):
        self.N += cluster.N
        
        for key in cluster.linear_sum.keys():
            try:
                self.linear_sum[key]  += cluster.linear_sum[key]
                self.squared_sum[key] += cluster.squared_sum[key]
            except KeyError:
                self.linear_sum[key]  = cluster.linear_sum[key]
                self.squared_sum[key] = cluster.squared_sum[key]
                
        if self.last_edit_time < cluster.creation_time:
            self.last_edit_time = cluster.creation_time
            
    def fading_function(self, time):
        return 2 ** (-self.decaying_factor * time)
    
    def setVertexRepresentative(self, v : Vertex): 
        self.m_vertexRepresentative = v

    def getVertexRepresentative(self):
        return self.m_vertexRepresentative
    
    def getStaticCenter(self):
        return self.m_staticCenter

    def setStaticCenter(self, timestamp):
        self.m_staticCenter = self.getCenter(timestamp).copy()
        
    def hasCenterChanged(self,percentage, refEpsilon, timestamp):
        distance = self.getCenterDistance(self.m_staticCenter, timestamp)
        if(distance > percentage * refEpsilon):
            return True
        return False
    def getCenterDistance(self, instance, timestamp):
        distance = 0.0
        center = self.getCenter(timestamp)
        for i in range(len(instance)):
            d = center[i] - instance[i]
            distance += d * d
        return math.sqrt(distance)
        
class Neighbour():
    def __init__(self, vertex = Vertex, dist=None):
        if dist is not None:
            self.m_coreDist = dist
        if vertex is not None:
            self.m_vertex   = vertex
            self.m_coreDist = dist

    def getDistance(self):
        return self.m_coreDist

    def String(self):
        return "value = {:.2f}".format(self.m_coreDist)

    def getVertex(self):
        return self.m_vertex
    
    def setCoredist(self, nn_dist):
        self.m_coreDist += nn_dist


class Edge():

    def __init__(self, v1 : Vertex, v2 : Vertex, dist : float):
        self.m_vertex1 = v1
        self.m_vertex2 = v2
        self.m_weight  = dist

    def compareTo(self, other):
        return self.m_weight < other.m_weight

    def getWeight(self):
        return self.m_weight

    def getVertex1(self):
        return self.m_vertex1

    def getVertex2(self):
        return self.m_vertex2

    def getAdjacentVertex(self, v):
        if v != self.m_vertex1:
            return self.m_vertex1
        else:
            return self.m_vertex2

    def setVertex1(self, v : Vertex):
        self.m_vertex1 = v

    def setVertex2(self, v : Vertex):
        self.m_vertex2 = v

    def setVertices(self, v1, v2):
        self.m_vertex1 = v1
        self.m_vertex2 = v2

    def graphVizString(self):
        return "M.add_edge(\"" + self.m_vertex1.getGraphVizVertexString() + "\",\"" + self.m_vertex2.getGraphVizVertexString() + "\",weight= " + str(self.m_weight) +")"

    def setEdgeWeight(self, weight):
        self.m_weight = weight


class SuperAdjacencyList:
    def __init__(self):
        self.m_adjacencyList = defaultdict(list)

    def addEdge(self, vertex, edge):
        
        if vertex not in self.m_adjacencyList:
            self.m_adjacencyList[vertex] = deque()
            self.m_adjacencyList[vertex].append(edge)
            return
        edges = self.m_adjacencyList[vertex]
        if (edge.getWeight() < edges[0].getWeight()):
            edges.appendleft(edge);
        else:
            edges.append(edge);
            

    def removeEdge(self, vertex):
        del self.m_adjacencyList[vertex]

    def getAdjacentVertices(self):
        return self.m_adjacencyList.keys()

    def getEdgeWithSmallestWeight(self, vertex):
        edges = self.m_adjacencyList[vertex]
        if not edges:
            return None
        res = min(edges, key=lambda edge: edge.getWeight())
        return res

    def getAdjacentEdges(self):
        all_edges = []
        for edges in self.m_adjacencyList.values():
            all_edges.extend(edges)
        return all_edges

    def getEdgesTo(self, vertex):
        return self.m_adjacencyList[vertex]

    def clear(self):
        self.m_adjacencyList = defaultdict(list)
        
class SuperAbstractGraph:
    def __init__(self):
        self.m_graph = defaultdict(SuperAdjacencyList)
        self.m_globalIDCounter = 0

    def addVertex(self, vertex):
        if vertex in self.m_graph:
            return False
        self.m_graph[vertex] = SuperAdjacencyList()
        return True

    def addEdge(self, vertex1, vertex2, edge):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise KeyError("One vertex or both are missing")
        self.adjacencyList(vertex1).addEdge(vertex2, edge)
        self.adjacencyList(vertex2).addEdge(vertex1, edge)

    def removeEdge(self, vertex1, vertex2):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise KeyError("One vertex or both are missing")
        self.adjacencyList(vertex1).removeEdge(vertex2)
        self.adjacencyList(vertex2).removeEdge(vertex1)

    def removeVertex(self, vertex):
        del self.m_graph[vertex]

    def buildGraph(self):
        pass

    def getEdge(self, vertex1, vertex2):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise KeyError("One vertex or both are missing")
        edges = self.adjacencyList(vertex1).getEdgesTo(vertex2)
        if edges is None or len(edges) == 0:
            raise KeyError("There are no edges between these vertices.")
        return edges

    def getVertices(self):
        if self.isEmpty():
            return None
        return self.m_graph.keys()

    def getAdjacentEdges(self, vertex):
        adjacentEdges = self.adjacencyList(vertex).getAdjacentEdges()
        if adjacentEdges is None:
            raise KeyError("This is an isolated vertex")
        return adjacentEdges

    def containsVertex(self, vertex):
        return vertex in self.m_graph

    def containsEdge(self, vertex1, vertex2):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise KeyError("One vertex or both are missing")
        return self.adjacencyList(vertex1).getEdgesTo(vertex2) is not None

    def __iter__(self):
        return iter(self.m_graph.keys())

    def numVertices(self):
        return len(self.m_graph)

    def isEmpty(self):
        return len(self.m_graph) == 0

    def getNextID(self):
        self.m_globalIDCounter += 1
        return self.m_globalIDCounter

    def adjacencyList(self, vertex):
        return self.m_graph[vertex]
    
class SuperCompleteGraph(SuperAbstractGraph):
    def __init__(self, superVertices, sourcegraph):
        super().__init__()
        self.m_supervertices = list(superVertices)
        self.m_sourcegraph = sourcegraph

    def buildGraph(self):
        for i in range(len(self.m_supervertices)):
            sv_i = self.m_supervertices[i]
            if not self.containsVertex(sv_i):
                self.addVertex(sv_i)
            for j in range(i + 1, len(self.m_supervertices)):
                sv_j = self.m_supervertices[j]
                if not self.containsVertex(sv_j):
                    self.addVertex(sv_j)
                min_edge = None
                dist = float('inf')
                for v in sv_i.getVertices():
                    for u in sv_j.getVertices():
                        e = self.m_sourcegraph.getEdge(v, u)
                        
                        w = 0
                        
                        if type(e.getWeight()) == float:
                            w = e.getWeight()
                        else:
                            w = e.getWeight().getWeight()

                        if w < dist:
                            min_edge = self.m_sourcegraph.getEdge(v, u)
                            dist = e.getWeight()
                assert min_edge is not None
                self.addEdge(sv_i, sv_j, min_edge) 
                
class AbstractGraph():
    def __init__(self):
        self.m_graph = {}
        self.m_globalIDCounter = 0

    def addVertex(self, vertex):
        if vertex in self.m_graph:
            return False
        self.m_graph[vertex] = {}
        return True

    def addEdge(self, vertex1 : Vertex, vertex2: Vertex, edge_weight):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise Exception("One vertex or both are missing")
        
        edge = None    
        
        for key, value in self.m_graph[vertex2].items():
            if key == vertex1:
                edge = Edge(vertex1, vertex2, edge_weight)
                break
        
        if edge is None:
            edge = Edge(vertex1, vertex2, edge_weight)
        
        self.addEdge1(vertex1, vertex2, edge)

    def addEdge1(self, vertex1, vertex2, edge : Edge):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise Exception("One vertex or both are missing")
        
        self.m_graph[vertex1][vertex2] = edge
        self.m_graph[vertex2][vertex1] = edge

    def removeEdge(self, vertex1, vertex2):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise Exception("One vertex or both are missing")
        
        del(self.m_graph[vertex1][vertex2])
        del(self.m_graph[vertex2][vertex1])

    def removeEdge2(self, edge):
        self.removeEdge(edge.getVertex1(), edge.getVertex2())

    def removeVertex(self, vertex):
        del self.m_graph[vertex]

    def buildGraph(self):
        pass

    def getEdge(self, vertex1, vertex2):
        if vertex1 not in self.m_graph or vertex2 not in self.m_graph:
            raise Exception("One vertex or both are missing")
        
        for v,w in self.adjacencyList(vertex1).items():
            if v == vertex2:
                return w
        
        return None

    def getVertices(self):
        return self.m_graph.keys()

    def getEdges(self):
        edges = set()
        for v in self.getVertices():
            for e in self.adjacencyList(v).values():
                edges.add(e)
        return edges

    def getAdjacentEdges(self, vertex):
        return self.m_graph[vertex]
    
    def containsVertex(self, vertex):
        return vertex in self.m_graph

    def containsEdge(self, vertex1, vertex2):
        if not self.containsVertex(vertex1) or not self.containsVertex(vertex2):
            raise Exception("One vertex or both are missing")
        for v in self.adjacencyList(vertex1).keys():
            if v == vertex2:
                return True                
        return False
    
    def containsEdge2(self, edge : Edge):
        if (self.containsVertex(edge.getVertex1()) and self.containsVertex(edge.getVertex2())):
            return self.containsEdge(edge.getVertex1(), edge.getVertex2())
        return False

    def __iter__(self):
        return iter(self.m_graph)

    def numVertices(self):
        return len(self.m_graph)

    def isEmpty(self):
        return not bool(self.m_graph)

    def getNextID(self):
        self.m_global_id_counter += 1
        return self.m_global_id_counter

    def adjacencyList(self, vertex):
        return self.m_graph[vertex]

    def getGraphVizString(self):
        edges = set()

        vertices = sorted(self.m_graph, key=lambda x: x.id)

        sb = []
        sb.append("graph {\n")

        for v in vertices:
            sb.append("\t" + v.get_graph_viz_string() + "\n")
            edges.update(self.adjacency_list(v).values())

        edges_sorted = sorted(edges, key=lambda x: (x.v1.id, x.v2.id))

        for e in edges_sorted:
            sb.append("\t" + e.graph_viz_string() + "\n")

        sb.append("}")
        return "".join(sb)

    def getAdjacencyMatrixAsArray(self):
        matrix = [[0.0 for _ in range(len(self.m_graph))] for _ in range(len(self.m_graph))]
        df = "{:.4f}"

        sorted_by_id = sorted(self.m_graph, key=lambda x: x.id)

        for row in range(len(sorted_by_id)):
            for column in range(len(sorted_by_id)):
                v1   = sorted_by_id[row]
                v2   = sorted_by_id[column]
                edge = self.m_graph[v1].get_edge_to(v2)
                
                if edge:
                    matrix[row][column] = edge.weight

        return matrix

    def extendWithSelfEdges(self):
        for v in self.m_graph:
            self_loop = Edge(v, v, v.getCoreDistance())
            self.addEdge(v, v, self_loop)
    
    def controlNumEdgesCompleteGraph(self):
        vertex_iterator = iter(self)
        edges           = set()
        
        for v in vertex_iterator:
            edges.update(self.adjacencyList(v).values())
            
        return len(edges) == int(self.numVertices() * (self.numVertices() - 1) / 2)
    
    #aqui pode ter um possível erro preciso revisar
    def hasSelfLoop(self, vertex: Vertex):
        if vertex not in self.m_graph:
            return Exception("Vertex does not exist!")
        
        return  vertex in self.adjacencyList(vertex).keys()
    
    def clearAdjacencyLists(self):
        for v in self.getVertices():
            self.m_graph[v].clear()
        
class MutualReachabilityGraph(AbstractGraph):
    def __init__(self, G, mcs : MicroCluster, minPts, timestamp):
        super().__init__()
        self.m_minPts  = minPts
        self.G         = G
        self.timestamp = timestamp

        for mc in mcs:
            v = Vertex(mc, timestamp)
            mc.setVertexRepresentative(v)
            self.G.add_node(v)
        
        start = time.time()
        self.computeCoreDistance(G, minPts)
        end   = time.time()
        #print(">tempo para computar coreDistanceDB",end - start, end='\n')

    def getKnngGraph(self):
        return self.knng
       
    def buildGraph(self):
        for v1 in self.G:            
            for v2 in self.G:
                if v1 != v2:
                    mrd = self.getMutualReachabilityDistance(v1, v2)
                    self.G.add_edge(v1, v2, weight = mrd)
        
        self.buildGraph1()
        
    def buildGraph1(self):
        for i, (u,v,w) in enumerate(self.G.edges(data='weight')):
            self.addVertex(u)
            self.addVertex(v)
            self.addEdge(u,v,w)
            
    def computeCoreDistance(self, vertices, minPts):
        for current in vertices:
            neighbours      = self.getNeighbourhood(current, vertices)
            minPtsNeighbour = neighbours[minPts - 1]
            
            current.setCoreDistance(minPtsNeighbour)

    def getNeighbourhood(self, vertex, vertices):
        neighbours = []
        
        for v in vertices:
            if v != vertex:
                neighbour = Neighbour(v, vertex.getDistance(v))
                neighbours.append(neighbour)
                
        neighbours.sort(key=lambda x: x.getDistance(), reverse=False)
        
        return neighbours

    def getMutualReachabilityDistance(self, v1, v2):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))
    
    def getMinPts(self):
        return self.m_minPts
        

class MinimalSpaningTree(AbstractGraph):
    def __init__(self, graph):
        super().__init__()
        self.m_inputGraph = graph

    def buildGraph(self):
        for i, (u,v,w) in enumerate(self.m_inputGraph.edges(data='weight')):
            self.addVertex(u)
            self.addVertex(v)
            self.addEdge(u,v,w)
    
    def getEdgeWithMinWeight(self, available):
        fromVertex = toVertex = edge = None
        dist = float('inf')
        
        for v in available:            
            for e in self.m_inputGraph.adjacencyList(v).values():
                other = e.getAdjacentVertex(v)
                if e.getWeight() < dist and other not in available:
                    fromVertex = v
                    toVertex = other
                    edge = e
                    dist = e.getWeight()

        return fromVertex, toVertex, edge

    @staticmethod
    def getEmptyMST():
        return MinimalSpaningTree(None)

    def getTotalWeight(self):
        edges = set()
        
        for v in self.m_graph.getVertices():
            for e in self.adjacencyList[v].values():
                edges.add(e)

        res = 0
        for e in edges:
            res += e.getWeight()
            
        return res
    
class SuperMinimalSpaningTree(MinimalSpaningTree):

    def __init__(self, graph):
        super().__init__(graph)
        self.m_inputGraph = graph

    def buildGraph(self):
        vertexQueue = set()
        edgeQueue = []
        
        # Select any node as the first node
        iterator = iter(self.m_inputGraph)
        first = None
        if iterator:
            first = next(iterator)
        else:
            return

        vertexQueue.add(first)

        while len(vertexQueue) != len(self.m_inputGraph.getVertices()):
            fromVertex, toVertex, edge = self.getEdgeWithMinWeight(vertexQueue)
            edgeQueue.append(edge)
            vertexQueue.add(toVertex)

        for sv in vertexQueue:
            for v in sv.getVertices():
                self.addVertex(v)

        for edge in edgeQueue:
            vertex1 = edge.getVertex1()
            vertex2 = edge.getVertex2()
            self.addEdge(vertex1, vertex2, edge)

        for sv in self.m_inputGraph.getVertices():
            c = sv.getComponent()
            edges = c.getEdges()
            for edge in edges:
                vertex1 = edge.getVertex1()
                vertex2 = edge.getVertex2()
                self.addEdge(vertex1, vertex2, edge)

    def getEdgeWithMinWeight(self, available):
        fromVertex = None
        toVertex = None
        edge = None
        dist = float('inf')

        for v in available:
            adjList = self.m_inputGraph.adjacencyList(v)
            adjVertices = adjList.getAdjacentVertices()

            for adjacentV in adjVertices:
                e = adjList.getEdgeWithSmallestWeight(adjacentV)
                if e.getWeight() < dist and adjacentV not in available:
                    fromVertex = v
                    toVertex = adjacentV
                    edge = e
                    dist = e.getWeight()

        return fromVertex, toVertex, edge

    @staticmethod
    def getEmptyMST():
        return SuperMinimalSpaningTree()
    
class Updating:
    def __init__(self, mrg: MutualReachabilityGraph, mst : MinimalSpaningTree):
        self.m_mrg = mrg
        self.m_mst = mst
        self.m_globalReplacementEdge = None
    
    def getMST(self):
        return self.m_mst
    
    def getMRG(self):
        return self.m_mrg
    
    def insert(self, mc):
        # Create new vertex for the new microcluster and insert it to the MRG
        insert = Vertex(mc)

        # Update the "adjacency matrix" of the mutual reachability graph
        # Furthermore generate candidates whose core-distance has changed and update them
        updatedObjects = self.computeAndCheckCoreDistance(insert, self.m_mrg.getVertices(), self.m_mrg.getMinPts())

        # Edge insertion i.e. edge's weight has decreased
        vertices = self.m_mrg.getVertices()
        # Update the corresponding row in the adjacency matrix
        for w in updatedObjects:
            for adj in self.m_mrg.adjacencyList(w).keys():
                max = self.m_mrg.getMutualReachabilityDistance(w, adj)
                if w != adj:
                    self.m_mrg.getEdge(w, adj).setEdgeWeight(max)

            # Fake vertex
            self.edgeInsertionMST(w)

        # Update the Mutual Reachability Graph, i.e. add new Vertex and the corresponding edges
        self.m_mrg.addVertex(insert)
        self.updateMRG_Ins(insert)

        # Vertex insertion into MST
        self.vertexInsertionMST(insert)
        return self.m_mst
    
    def computeAndCheckCoreDistance(self, vertex, vertices, minPts):
        vertexList = [vertex]
        candidateSet = []
        neighbours = []
        
        
        self.computeNeighbourhoodAndCandidates(vertex, neighbours, candidateSet, vertexList, minPts)
        neighbours.sort(key=lambda n: n.getDistance(), reverse = False)
        self.computeNeighbourhoodAndCandidates(vertex, neighbours, candidateSet, vertices, minPts)
        neighbours.sort(key=lambda n: n.getDistance(), reverse = False)

        coreDist = neighbours[len(neighbours) - 1]
        vertex.setCoreDistance(coreDist)

        i = 0
        while i < len(candidateSet):            
            candidate = candidateSet[i]
            neighbours = []
            self.computeNeighbourhoodAndCandidates(candidate, neighbours, None, vertexList, minPts)
            neighbours.sort(key=lambda n: n.getDistance(), reverse = False)
            self.computeNeighbourhoodAndCandidates(candidate, neighbours, None, vertices, minPts)
            neighbours.sort(key=lambda n: n.getDistance(), reverse = False)

            coreDist = neighbours[len(neighbours) - 1]
            if coreDist.getDistance() < candidate.getCoreDistance():
                candidate.setCoreDistance(coreDist)
                i += 1
            else:
                del candidateSet[i]

        return candidateSet
    
    def computeNeighbourhoodAndCandidates(self, vertex, lista, candidateSet, vertices, minPts):
        k = minPts - 1
        for v in vertices:
            dist = vertex.getDistance(v)
            if candidateSet is not None and dist < v.getCoreDistance():
                candidateSet.append(v)

            if len(lista) < minPts:
                lista.append(Neighbour(v, dist))
            else:
                kth = lista[k]
                if dist < kth.getDistance():
                    lista.append(Neighbour(v, dist))
                    if kth.getDistance() != lista[k].getDistance():
                        for last in range(len(lista) - 1, k, -1):
                            del lista[last]
    
    def edgeInsertionMST(self, w):
        # Generate the fake vertex to insert
        z = Vertex(None, -1)
        self.m_mrg.addVertex(z)
        # Create fake vertex with edges with the weights from MRG
        edge = None
        assert len(self.m_mst.getVertices()) == len(self.m_mrg.getVertices()) - 1
        for v in self.m_mst.getVertices():
            if v == w:
                edge = Edge(z, v, -1)
            else:
                edge = Edge(z, v, self.m_mrg.getEdge(w, v).getWeight())

            assert edge is not None
            self.m_mrg.addEdge(z, v, edge)

        newMSTedges = []
        self.m_globalReplacementEdge = None

        # Get a random vertex to represent the root of the MST
        first = None
        iterator = iter(self.m_mst)
        if iterator.hasNext():
            first = iterator.next()
            if first == w and iterator.hasNext():
                first = iterator.next()
            else:
                # This can only happen when the MRG was empty and a vertex was added.
                # Since that vertex is the only existing vertex in the MRG, it represents the MST
                # TODO: Single insertion of vertex in empty MST
                pass

        assert first != w

        for v in self.m_mrg.getVertices():
            v.resetVisited()

        self.updateMST_Ins(first, z, newMSTedges)
        newMSTedges.append(self.m_globalReplacementEdge)

        assert len(newMSTedges) == len(self.m_mst.getVertices())

        # Update MST adjacency lists
        self.m_mst.clearAdjacencyLists()

        correctMSTedges = []
        for e in newMSTedges:
            vertex1 = e.getVertex1()
            vertex2 = e.getVertex2()
            if vertex1 != z and vertex2 != z:
                self.m_mst.addEdge(vertex1, vertex2, e)
                correctMSTedges.append(e)
            else:
                v = e.getAdjacentVertex(z)
                if v == w:
                    # Do nothing --> remove the edge to the fake vertex
                    pass
                else:
                    replacement = self.m_mrg.getEdge(w, v)
                    assert replacement.getWeight() == e.getWeight()
                    self.m_mst.addEdge(w, v, replacement)
                    correctMSTedges.append(replacement)

        assert len(correctMSTedges) == len(self.m_mst.getVertices()) - 1

        # Remove fake node from m_mrg
        for v in self.m_mrg.getVertices():
            self.m_mrg.removeEdge(z, v)
        self.m_mrg.removeVertex(z)

        assert not self.m_mrg.containsVertex(z)
        assert not self.m_mst.containsVertex(z)
    
    def updateMST_Ins(self, r, z, edges):
        r.setVisited()
        m = self.m_mrg.getEdge(z, r)  # z.getEdgeTo(r)
        adjacentVertices = self.m_mst.adjacencyList(r).keys()
        for w in adjacentVertices:
            if not w.visited():
                self.updateMST_Ins(w, z, edges)
                wr = self.m_mrg.getEdge(w, r)  # w.getEdgeTo(r)
                k = None
                h = None
                aux = 0
                aux3 = 0
                
                if type(self.m_globalReplacementEdge.getWeight()) == float:
                    aux = self.m_globalReplacementEdge.getWeight()
                else:
                    aux = self.m_globalReplacementEdge.getWeight().getWeight()
                    
                if type(wr.getWeight()) == float:
                    aux3 = wr.getWeight()
                else:
                    aux3 = wr.getWeight().getWeight()
                
                if aux > aux3:
                    k = self.m_globalReplacementEdge
                    h = wr
                else:
                    k = wr
                    h = self.m_globalReplacementEdge
                edges.append(h)
                
                
                
                if type(k.getWeight()) == float:
                    aux1 = k.getWeight()
                else:
                    aux1 = k.getWeight().getWeight()
                
                if type(m.getWeight()) == float:
                    aux2 = m.getWeight()
                else:
                    aux2 = m.getWeight().getWeight()
                    
                if aux1 < aux2:
                    m = k
        self.m_globalReplacementEdge = m

    def updateMRG_Ins(self, vertex):
        vertices = self.m_mrg.getVertices()

        for v in vertices:
            # No self loops
            if vertex != v:
                max = self.m_mrg.getMutualReachabilityDistance(vertex, v)
                edge = Edge(vertex, v, max)
                self.m_mrg.addEdge(vertex, v, edge)
                self.m_mrg.addEdge(v, vertex, edge)

        assert self.m_mrg.controlNumEdgesCompleteGraph()
    
    def vertexInsertionMST(self, insert):
        newMSTedges = []
        self.m_globalReplacementEdge = None

        first = None
        iterator = iter(self.m_mst)
        if next(iterator):
            first = next(iterator)
            if first == insert and iterator.hasNext():
                first = iterator.next()
            else:
                # TODO: Handle the case when the MRG was empty and a vertex was added
                pass
        assert first != insert

        for v in self.m_mrg.getVertices():
            v.resetVisited()

        self.updateMST_Ins(first, insert, newMSTedges)
        newMSTedges.append(self.m_globalReplacementEdge)

        assert len(newMSTedges) == len(self.m_mst.getVertices())

        self.m_mst.clearAdjacencyLists()
        self.m_mst.addVertex(insert)

        for e in newMSTedges:
            vertex1 = e.getVertex1()
            vertex2 = e.getVertex2()
            self.m_mst.addEdge(vertex1, vertex2, e)
    
    def getAffectedNeighborhood2(self, query, vertices):
        neighbors = set()
        dist = -1.0
        for v in vertices:
            dist = v.getDistance(query)
            if v.getCoreDistance() >= dist:
                neighbors.add(v)
        return neighbors
    def getAffectedNeighborhood(self, vertex):
        affectedNeighbours = set()

        affectedNeighbours.update(self.getAffectedNeighborhood2(vertex, self.m_mrg.getVertices()))
        
        if vertex in affectedNeighbours:
            affectedNeighbours.remove(vertex)
            print(vertex)
        
        

        queue = []
        queue.extend(affectedNeighbours)

        while queue:
            first = queue.pop(0)

            neighbours = self.getAffectedNeighborhood2(first, self.m_mrg.getVertices())
            neighbours.remove(vertex)

            for v in neighbours:
                if v not in affectedNeighbours:
                    affectedNeighbours.add(v)
                    queue.append(v)

        assert vertex not in affectedNeighbours

        return affectedNeighbours
                                  
    def delete(self, mc):
        assert mc.getVertexRepresentative() != None, "Vertex reference is missing"
        toDelete = mc.getVertexRepresentative()

        if len(self.m_mrg.getVertices()) - 1 <= self.m_mrg.getMinPts():
            self.m_mst = None
            return None

        affectedNeighbours = self.getAffectedNeighborhood(toDelete)

        self.updateMRG_Del(toDelete)
        for affected in affectedNeighbours:
            self.updateCoreDistance(affected, self.m_mrg.getVertices(), self.m_mrg.getMinPts())

        for affected in affectedNeighbours:
            edges = self.m_mrg.getAdjacentEdges(affected)
            for e in edges:
                adjacent = e.getAdjacentVertex(affected)
                max_val = self.m_mrg.getMutualReachabilityDistance(affected, adjacent)
                if e.getWeight() < max_val:
                    e.setEdgeWeight(max_val)

        for w in affectedNeighbours:
            for adj in self.m_mrg.adjacencyList(w).keys():
                max_val = self.m_mrg.getMutualReachabilityDistance(w, adj)
                if w != adj:
                    self.m_mrg.getEdge(w, adj).setEdgeWeight(max_val)

        mst_components = self.removeFromMST_Del(toDelete, affectedNeighbours)

        superVertices = set()
        for c in mst_components:
            superVertices.add(SuperVertex(c))

        assert len(mst_components) == len(superVertices)

        self.m_mst = self.updateMST_Del(superVertices)

        return self.m_mst
                                  
    def updateMRG_Del(self, v):
        toRemove = set()
        toRemove.update(self.m_mrg.adjacencyList(v).values())
        for edge in toRemove:
            self.m_mrg.removeEdge2(edge)
        self.m_mrg.removeVertex(v)

    def updateCoreDistance(self, vertex, vertices, minPts):
        lista = []

        dist = -1.0
        for v in vertices:
            dist = vertex.getDistance(v)
            lista.append(Neighbour(v, dist))

        assert lista.size() == len(vertices)

        coreDist = lista[minPts - 1]
        vertex.setCoreDistance(coreDist)
    
    def removeFromMST_Del(self, toDelete, affectedNeighbours):
        components = set()
        startVertices = set()

        toRemove = set()
        print(self.m_mst.getAdjacentEdges(toDelete).values())
        toRemove.update(self.m_mst.getAdjacentEdges(toDelete).values())

        for edge in toRemove:
            self.m_mst.removeEdge2(edge)
            startVertices.add(edge.getAdjacentVertex(toDelete))
        self.m_mst.removeVertex(toDelete)

        for v in affectedNeighbours:
            toRemove.clear()
            toRemove.update(self.m_mst.getAdjacentEdges(v))

            adjVertices = set()
            for edge in toRemove:
                adj = edge.getAdjacentVertex(v)
                adjVertices.add(adj)
                self.m_mst.removeEdge2(edge)

            for adj in adjVertices:
                startVertices.add(adj)

            startVertices.add(v)

        for startVertex in startVertices:
            newComponent = True
            for c in components:
                if c.containsVertex(startVertex):
                    newComponent = False
                    break
            if newComponent:
                debug = Component(startVertex, self.m_mst, True)
                components.add(debug)

        self.assertNumVertices(components, self.m_mst.numVertices())

        return components

    def assertNumVertices(self, components, numVertices):
        sum_numVertices = 0
        pairwiseDifferent = True
        for c in components:
            sum_numVertices += c.numVertices()
            for other in components:
                if c != other:
                    pairwiseDifferent &= not c.compareByVertices(other)

        return pairwiseDifferent and sum_numVertices == numVertices
    
    def updateMST_Del(self, superVertices):
        scgraph = SuperCompleteGraph(superVertices, self.m_mrg)
        scgraph.buildGraph()
        new_mst = SuperMinimalSpaningTree(scgraph)
        new_mst.buildGraph()
        return new_mst
                                  
class Component(AbstractGraph):
    def __init__(self, startVertex: Vertex, graph: AbstractGraph, prepareEdges: bool):
        super().__init__()     
        self.m_prepare_edges = prepareEdges
        self.m_edges_summarized_by_weight = {}
        
        self.addVertex(startVertex)
        if graph.hasSelfLoop(startVertex):
            self.addEdge(startVertex, startVertex, graph.getEdge(startVertex, startVertex))
            
        self.build(startVertex, graph)

    def build(self, vertex: Vertex, graph: AbstractGraph):
        adjacentVertices = graph.adjacencyList(vertex).keys()
        
        for v in adjacentVertices:
            if not super().containsVertex(v):
                self.addVertex(v)
                
                if graph.hasSelfLoop(v):
                    self.addEdge(v, v, graph.getEdge(v, v))
                if not self.containsEdge(vertex, v):
                    self.addEdge(vertex, v, graph.getEdge(vertex, v))
                
                self.build(v, graph)

    def compareByVertices(self, other: "Component"):
        if self.numVertices() != other.numVertices():
            return False
        
        iterator = iter(self)
        
        for v in next(iterator):
            if v not in other.containsVertex(v):
                return False
        return True

    def buildGraph(self):
        pass
    
    def setMEdge(self, a):
        self.m_edges_summarized_by_weight = a
        
    def getMEdge(self):
        return self.m_edges_summarized_by_weight

    def split(self, e: Edge):
        self.removeEdge(e.getVertex1(), e.getVertex2())
        a   = Component(e.getVertex1(), self)
        b   = Component(e.getVertex2(), self)
        res = set()
        res.add(a)
        res.add(b)
        return res

class Node:
    s_label = 0

    def __init__(self, c):
        self.m_vertices = set(c)
        self.m_children = []
        self.m_delta = True
        self.m_label = Node.s_label
        Node.s_label += 1
        self.m_parent = None
        self.m_scaleValue = 0
        

    def computeStability(self) -> float:
        if self.m_parent is None:
            return float('nan')

        eps_max = self.m_parent.m_scaleValue
        eps_min = self.m_scaleValue        
        
        if eps_max == 0:
            eps_max = 0.0000000001
        if eps_min == 0:
            eps_min = 0.0000000001
        
        self.m_stability = len(self.m_vertices) * ((1 / eps_min) - (1 / eps_max))

        return self.m_stability

    def addChild(self, child: "Node"):
        self.m_children.append(child)

    def getChildren(self):
        return self.m_children

    def setParent(self, parent):
        self.m_parent = parent

    def getParent(self):
        return self.m_parent

    def setScaleValue(self, scaleValue):
        self.m_scaleValue = scaleValue

    def getScaleValue(self):
        return self.m_scaleValue

    def getVertices(self):
        return self.m_vertices

    def setDelta(self):
        self.m_delta = True

    def resetDelta(self):
        self.m_delta = False

    def isDiscarded(self):
        return not self.m_delta

    def getStability(self) -> float:
        return self.m_stability

    def getPropagatedStability(self) -> float:
        return self.m_propagatedStability

    def setPropagatedStability(self, stability):
        self.m_propagatedStability = stability

    @staticmethod
    def resetStaticLabelCounter():
        Node.s_label = 0

    #def __str__(self):
    #    return self.getDescription()

    def getDescription(self):
        return f'N={len(self.m_vertices)},SV={self.m_scaleValue},SC={self.m_stability}'

    def getOutputDescription(self):
        return f'{len(self.m_vertices)},{self.m_scaleValue},{self.m_stability}'

    def getGraphVizNodeString(self):
        return f'node{self.m_label}'

    def getGraphVizEdgeLabelString(self):
        return f'[label="{self.m_scaleValue}"];'

    def getGraphVizString(self):
        return f'{self.getGraphVizNodeString()} [label="Num={len(self.m_vertices)}[SV,SC,D]:{{{self.m_scaleValue}; {self.m_stability}; {self.m_delta}}}""];'

    def setVertices(self, vertices ):
        self.m_vertices = set(vertices)

class DendrogramComponent(Component):
    def __init__(self, start_vertex: Vertex, graph: AbstractGraph, prepareEdges: bool):  
        
        super().__init__(start_vertex, graph, prepareEdges)
        
        self.m_set_of_highest_weighted_edges = set()
        self.m_prepare_edges = prepareEdges
        self.m_node = None

        self.addVertex(start_vertex)
        
        
        if graph.hasSelfLoop(start_vertex):
            self.addEdge(start_vertex, start_vertex, graph.getEdge(start_vertex, start_vertex))
        
            

        self.build(start_vertex, graph)
         

    def build(self, vertex: Vertex, graph: AbstractGraph):
        adjacent_vertices = graph.adjacencyList(vertex).keys()
        
        for v in adjacent_vertices:
            if not self.containsVertex(v):
                self.addVertex(v)

                if graph.hasSelfLoop(v):
                    self.addEdge(v, v, graph.getEdge(v, v))

                if not self.containsEdge(vertex, v):
                    edge = graph.getEdge(vertex, v)

                    self.addEdge(vertex, v, edge)

                    w = edge.getWeight()
                    if isinstance(w, Edge):
                        w = (edge.getWeight()).getWeight()

                    if self.m_prepare_edges:
                        
                        if w not in self.m_edges_summarized_by_weight:
                            self.m_edges_summarized_by_weight[w] = set()
                            

                        self.m_edges_summarized_by_weight[w].add(edge)
                
                self.build(v, graph)
        
    def setHeighestWeightedEdges(self):        
        if self.m_prepare_edges:
            highest = -1.0
            
            for weight in self.m_edges_summarized_by_weight.keys():
                
                if type(weight) == Edge:
                    weight = weight.getWeight()
                
                if weight > highest:
                    highest = weight
            
            
            if highest == -1:
                self.m_set_of_highest_weighted_edges = None
            else:        
                self.m_set_of_highest_weighted_edges = self.m_edges_summarized_by_weight[highest]
                del self.m_edges_summarized_by_weight[highest]

    def getNextSetOfHeighestWeightedEdges(self):
        if self.m_set_of_highest_weighted_edges is None or len(self.m_set_of_highest_weighted_edges) == 0:
            
            self.setHeighestWeightedEdges()
        
        
        res = self.m_set_of_highest_weighted_edges
        self.setHeighestWeightedEdges()  # prepare next step
        return res

    def splitComponent(self, e: Edge):
        self.removeEdge(e.getVertex1(), e.getVertex2())

        a = DendrogramComponent(e.getVertex1(), self, False)
        b = DendrogramComponent(e.getVertex2(), self, False)

        res = {a, b}
        return res

    def extendWithSelfEdges(self):
        for v in self.getVertices():
            self_loop = Edge(v, v, v.getCoreDistance())
            self.addEdge(v, v, self_loop)

            w = self_loop.getWeight()
            if w not in self.m_edges_summarized_by_weight:
                self.m_edges_summarized_by_weight[w] = set()

            self.m_edges_summarized_by_weight[w].add(self_loop)

    def setNodeRepresentitive(self, node: Node):
        self.m_node = node

    def getNode(self):
        return self.m_node
    def getMEdge(self):
        return self.m_edges_summarized_by_weight

    def String(self):
        sb = []

        for v in self.get_vertices():
            sb.append(str(v))

        return f"[{''.join(sb)}]"

class Dendrogram:
    def __init__(self, mst: MinimalSpaningTree, min_cluster_size: int, timestamp):        
        assert len(mst.getVertices()) > 0
        Node.resetStaticLabelCounter()

        self.m_components     = []
        self.m_minClusterSize = min_cluster_size
        first                 = None
        it                    = iter(mst.getVertices())
        self.timestamp        = timestamp
        
        if next(it):
            first = next(it)

        assert first is not None

        self.m_mstCopy = DendrogramComponent(first, mst, True)
        
        
        self.m_root = Node(self.m_mstCopy.getVertices())
        
        self.spurious_1 = 0
        self.spurious_gr2 = 0
        
        self.m_mstCopy.setNodeRepresentitive(self.m_root)
        self.m_components.append(self.m_mstCopy)

    def build(self):
        self.experimental_build()

    def compare(self, n1: Node):
        return n1.getScaleValue()

    def clusterSelection(self):
        selection = []

        # Step 1
        leaves = self.getLeaves(self.m_root)
        
        for leaf in leaves:
            leaf.setPropagatedStability(leaf.computeStability())

        # Special case
        if len(leaves) == 1 and leaves[0] == self.m_root:
            selection.append(self.m_root)
            return selection

        queue = []
        for leaf in leaves:
            if leaf.getParent() is not None and leaf.getParent() not in queue:
                queue.append(leaf.getParent())
        
        queue.sort(key=self.compare)

        # Step 2
        while queue and queue[0] != self.m_root:
            current = queue[0]
            current_stability = current.computeStability()
            s = sum(child.getPropagatedStability() for child in current.getChildren())
            if current_stability < s:
                current.setPropagatedStability(s)
                current.resetDelta()
            else:
                current.setPropagatedStability(current_stability)
            queue.remove(current)
            if current.getParent() not in queue and current.getParent() is not None:
                queue.append(current.getParent())
            queue.sort(key=self.compare)

        # get clustering selection
        selection_queue = self.m_root.getChildren().copy()
        self.m_root.resetDelta()

        while selection_queue:
            current = selection_queue.pop(0)
            if not current.isDiscarded():
                selection.append(current)
            else:
                selection_queue.extend(current.getChildren())

        return selection

    @staticmethod
    def getLeaves(node: Node):
        res = []
        queue = [node]

        while queue:
            n = queue.pop(0)

            if len(n.getChildren()) > 0:
                queue.extend(n.getChildren())
            elif len(n.getChildren()) == 0:
                res.append(n)
            #queue.pop(0)

        return res
    
    def experimental_build(self):
    
        # Get set of edges with the highest weight
        
        next = self.m_mstCopy.getNextSetOfHeighestWeightedEdges()

        # return if no edge available
        if next is None:
            return

        # repeat until all edges are processed
        while next is not None:
                        
            # copy edges into "queue"
            highestWeighted = []
            highestWeighted.extend(next)

            # Mapping of a component onto it's subcomponents, resulting from splitting
            splittingCandidates = {}

            # search components which contains one of the edges which the highest weight
            for edge in highestWeighted:
                i = 0
                while i < len(self.m_components):
                    current = self.m_components[i]

                    if current.containsEdge2(edge):
                        tmp = [current]
                        splittingCandidates[current] = tmp

                        self.m_components.pop(i)
                    else:
                        i += 1

            # Split these components
            for current in splittingCandidates.keys():  # Nodes
                if len(highestWeighted) == 0:
                    break

                currentNode = current.getNode()  # get the DendrogramComponent node to access the internal DBs
                
                highest = highestWeighted[0].getWeight()
                if isinstance(highest, Edge):
                    highest = (highestWeighted[0].getWeight()).getWeight()
                    
                
                current.getNode().setScaleValue(highest)  # epsilon
        
                subComponents = splittingCandidates[current]  # get TMP

                # Info: call by reference with "subComponent".
                # Effect: After calling splitting(), the subComponent List contains the subcomponents
                # which are created by removing the edges from the whole component
                toRemove = self.splitting(subComponents, highestWeighted)

                for item in toRemove:
                    highestWeighted.remove(item)

                spuriousList = []

                # Computing the Matrix Dh for HAI
                # computingMatrixDhHAI(numberDB, current, splittingCandidates.get(current));
                for c in splittingCandidates[current]:  # scrolls through the list of splitting Components (HAI)
                    numPoints = 0.0
                    count     = 0

                    for v in c.getVertices():
                        numPoints += v.getMicroCluster().getWeight(self.timestamp)
                        count     += 1

                    if numPoints >= self.m_minClusterSize or count >= self.m_minClusterSize:
                        spuriousList.append(c)

                if len(spuriousList) == 1:
                    self.spurious_1 += 1

                    # cluster has shrunk
                    replacementComponent = spuriousList.pop(0)

                    replacementComponent.setNodeRepresentitive(currentNode)

                    assert len(spuriousList) == 0

                    # add component to component list for further processing
                    self.m_components.append(replacementComponent)

                elif len(spuriousList) > 1:
                    
                    self.spurious_gr2 += 1

                    for c in spuriousList:
                        # generate new child node with currentNode as parent node
                        child = Node(c.getVertices())

                        child.setParent(currentNode)

                        # add child to parent
                        currentNode.addChild(child)
                        c.setNodeRepresentitive(child)

                        # add new components to component list for further processing
                        self.m_components.append(c)
                        
                        child.setScaleValue(highest)

            # update set of heighest edges
            next = self.m_mstCopy.getNextSetOfHeighestWeightedEdges()

    def splitting(self, c, edges):
        to_remove = []
        for edge in edges:
            i = 0
            while i < len(c):
                current = c[i]
                if current.containsEdge2(edge):
                    to_remove.append(edge)
                    v1, v2 = edge.getVertex1(), edge.getVertex2()
                    if v1 == v2:
                        current.removeEdge(edge)
                    else:
                        c.pop(i)
                        c.extend(current.splitComponent(edge))
                    break
                else:
                    i += 1
        return to_remove
            
class HDBStream(base.Clusterer):
    
    class BufferItem:
        def __init__(self, x, timestamp, covered):
            self.x         = x
            self.timestamp = (timestamp,)
            self.covered   = covered

    def __init__(
        self,
        m_minPoints            = 10,
        decaying_factor: float = 0.25,
        beta: float            = 0.75,
        mu: float              = 2,
        epsilon: float         = 0.02,
        n_samples_init: int    = 1000,
        stream_speed: int      = 3000,
        step                   = 1,
        m_movementThreshold    = 0.5,
        runtime                = False,
        plot                   = False,
        save_partitions        = False,
        percent                = 0.1,
        method_summarization   = 'single_linkage',
    ):
        super().__init__()
        self.percent              = percent
        self.timestamp            = 0
        self.initialized          = False
        self.decaying_factor      = decaying_factor
        self.beta                 = beta
        self.mu                   = mu
        self.epsilon              = epsilon
        self.n_samples_init       = n_samples_init
        self.stream_speed         = stream_speed
        self.mst                  = None
        self.mst_mult             = None
        self.m_minPoints          = m_minPoints
        self.step                 = step
        self.m_movementThreshold  = m_movementThreshold
        self.runtime              = runtime
        self.plot                 = plot
        self.save_partitions      = save_partitions
        self.method_summarization = method_summarization
        
        # number of clusters generated by applying the variant of DBSCAN algorithm
        # on p-micro-cluster centers and their centers
        self.n_clusters = 0
        
        self.clusters: typing.Dict[int, "MicroCluster"]         = {}
        self.p_micro_clusters: typing.Dict[int, "MicroCluster"] = {}
        self.o_micro_clusters: typing.Dict[int, "MicroCluster"] = {}
        
        #mudei o método pq estava dando erro e não estavamos precisando no momento
        self._time_period = math.ceil((1 / self.decaying_factor) * math.log((self.mu * self.beta) / (self.mu * self.beta - 1))) + 1
        
        print("self._time_period", self._time_period)
        self._init_buffer: typing.Deque[typing.Dict] = deque()
        
        self._n_samples_seen = 0
        self.m_update        = None

        # DataFrame to save the runtimes
        if self.runtime:
            self.df_runtime_final  = pd.DataFrame(columns=['timestamp', 'micro_clusters', 'summarization', 'multiple_hierarchies'])
            self.df_runtime        = pd.DataFrame(columns=['minpts', 'mrg', 'mst', 'dendrogram', 'selection', 'total'])

        # check that the value of beta is within the range (0,1]
        if not (0 < self.beta <= 1):
            raise ValueError(f"The value of `beta` (currently {self.beta}) must be within the range (0,1].")

    @property
    def centers(self):
        return {k: cluster.calc_center(self.timestamp) for k, cluster in self.clusters.items()}

    @staticmethod
    def _distance(point_a, point_b):
        square_sum = 0
        dim        = len(point_a)
        
        for i in range(dim):
            square_sum += math.pow(point_a[i] - point_b[i], 2)
        
        return math.sqrt(square_sum)
    
    def distanceEuclidian(self, x1, x2):
        distance = 0
        
        for i in range(len(x1)):
            d         = x1[i] - x2[i]
            distance += d * d
            
        return math.sqrt(distance)

    def _get_closest_cluster_key(self, point, clusters):
        min_distance = math.inf
        key          = -1
        
        for k, cluster in clusters.items():
            distance = self.distanceEuclidian(cluster.getCenter(self.timestamp), point)
            
            if distance < min_distance and distance <= self.epsilon:
                min_distance = distance
                key          = k
                
        return key

    def _merge(self, point):
        # initiate merged status
        merged_status = False

        pos = self._n_samples_seen - 1

        if len(self.p_micro_clusters) != 0:
            # try to merge p into its nearest p-micro-cluster c_p
            closest_pmc_key = self._get_closest_cluster_key(point, self.p_micro_clusters)
            
            if closest_pmc_key != -1:
                updated_pmc = copy.deepcopy(self.p_micro_clusters[closest_pmc_key])
                updated_pmc.insert(point, self.timestamp)
                
                if updated_pmc.getRadius(self.timestamp) <= self.epsilon:
                    # keep updated p-micro-cluster
                    self.p_micro_clusters[closest_pmc_key] = updated_pmc

                    df_mc_to_points.loc[pos, 'id_mc'] = closest_pmc_key

                    merged_status = True

        if not merged_status:
            closest_omc_key = self._get_closest_cluster_key(point, self.o_micro_clusters)
            
            if closest_omc_key != -1:
                updated_omc = copy.deepcopy(self.o_micro_clusters[closest_omc_key])
                updated_omc.insert(point, self.timestamp)

                if updated_omc.getRadius(self.timestamp) <= self.epsilon:
                    # keep updated o-micro-cluster
                    weight_omc = updated_omc.getWeight(self.timestamp)
                    
                    if weight_omc > self.mu * self.beta:
                        # it has grown into a p-micro-cluster
                        del self.o_micro_clusters[closest_omc_key]

                        new_key = 0
                        
                        while new_key in self.p_micro_clusters:
                            new_key += 1
                            
                        updated_omc.setID(new_key)
                        self.p_micro_clusters[new_key] = updated_omc

                        df_mc_to_points.loc[pos, 'id_mc'] = new_key
                        df_mc_to_points['id_mc']          = df_mc_to_points['id_mc'].replace((-1) * closest_omc_key, new_key)
                            
                    else:
                        self.o_micro_clusters[closest_omc_key] = updated_omc

                        # Outliers have our key negative
                        df_mc_to_points.loc[pos, 'id_mc'] = (-1) * closest_omc_key
                    
                    merged_status = True
                    
        if not merged_status:
            # create a new o-data_bubble by p and add it to o_data_bubbles
            mc_from_p = MicroCluster(x=point, timestamp=self.timestamp, decaying_factor=self.decaying_factor)

            key_o = 2

            while key_o in self.o_micro_clusters:
                key_o += 1

            self.o_micro_clusters[key_o] = mc_from_p

            df_mc_to_points.loc[pos, 'id_mc'] = (-1) * key_o

            merged_status = True
    
    def _merge_old(self, point):
        # initiate merged status
        merged_status = False
        
        pos = self._n_samples_seen - 1

        if len(self.p_micro_clusters) != 0:
            # try to merge p into its nearest p-micro-cluster c_p
            closest_pmc_key = self._get_closest_cluster_key(point, self.p_micro_clusters)
            
            if closest_pmc_key != -1:
                updated_pmc     = copy.deepcopy(self.p_micro_clusters[closest_pmc_key])
                updated_pmc.insert(point, self.timestamp)

                if updated_pmc.getRadius(self.timestamp) <= self.epsilon:
                    # keep updated p-micro-cluster
                    self.p_micro_clusters[closest_pmc_key] = updated_pmc
                    merged_status = True

                    df_mc_to_points.loc[pos, 'id_mc'] = closest_pmc_key

                    if self.p_micro_clusters[closest_pmc_key].hasCenterChanged(self.m_movementThreshold, self.epsilon, self.timestamp):
                        self.p_micro_clusters[closest_pmc_key].setStaticCenter(self.timestamp)

        if not merged_status and len(self.o_micro_clusters) != 0:
            
            closest_omc_key = self._get_closest_cluster_key(point, self.o_micro_clusters)
            
            if closest_omc_key != -1:
                updated_omc     = copy.deepcopy(self.o_micro_clusters[closest_omc_key])
                updated_omc.insert(point, self.timestamp)

                if updated_omc.getRadius(self.timestamp) <= self.epsilon:
                    # keep updated o-micro-cluster
                    if updated_omc.getWeight(self.timestamp) > self.mu * self.beta:
                        # it has grown into a p-micro-cluster
                        del self.o_micro_clusters[closest_omc_key]
                        updated_omc.setStaticCenter(self.timestamp)

                        new_key = 0

                        while new_key in self.p_micro_clusters:
                            new_key += 1

                        updated_omc.setID(new_key)
                        self.p_micro_clusters[new_key] = updated_omc

                        df_mc_to_points.loc[pos, 'id_mc'] = new_key
                        df_mc_to_points['id_mc']          = df_mc_to_points['id_mc'].replace((-1) * closest_omc_key, new_key)

                    else:
                        self.o_micro_clusters[closest_omc_key] = updated_omc                    

                        # Outliers have our key negative
                        df_mc_to_points.loc[pos, 'id_mc'] = (-1) * closest_omc_key

                    merged_status = True
                
        if not merged_status:
            # create a new o-data_bubble by p and add it to o_micro_clusters
            mc_from_p = MicroCluster(x=point, timestamp=self.timestamp, decaying_factor=self.decaying_factor)

            key_o = 2

            while key_o in self.o_micro_clusters:
                key_o += 1

            self.o_micro_clusters[key_o] = mc_from_p

            df_mc_to_points.loc[pos, 'id_mc'] = (-1) * key_o

            merged_status = True

        # when p is not merged and o-micro-cluster set is empty
        # if not merged_status and len(self.o_micro_clusters) == 0:
        #    mc_from_p = MicroCluster(x=point, timestamp=self.timestamp, decaying_factor=self.decaying_factor)
        #    self.o_micro_clusters = {2: mc_from_p}
        #    df_mc_to_points.loc[pos, 'id_mc'] = -2
        #    merged_status = True
            
    def _is_directly_density_reachable(self, c_p, c_q):
        if c_p.calc_weight() > self.mu and c_q.calc_weight() > self.mu:
            # check distance of two clusters and compare with 2*epsilon
            c_p_center = c_p.calc_center(self.timestamp)
            c_q_center = c_q.calc_center(self.timestamp)
            distance   = self._distance(c_p_center, c_q_center)
            
            if distance < 2 * self.epsilon and distance <= c_p.calc_radius() + c_q.calc_radius():
                return True
        return False

    def _query_neighbor(self, cluster):
        neighbors = deque()
        # scan all clusters within self.p_micro_clusters
        for pmc in self.p_micro_clusters.values():
            # check density reachable and that the cluster itself does not appear in neighbors
            if cluster != pmc and self._is_directly_density_reachable(cluster, pmc):
                neighbors.append(pmc)
        return neighbors

    @staticmethod
    def _generate_clusters_for_labels(cluster_labels):
        # initiate the dictionary for final clusters
        clusters = {}

        # group clusters per label
        mcs_per_label = defaultdict(deque)
        for mc, label in cluster_labels.items():
            mcs_per_label[label].append(mc)

        # generate set of clusters with the same label
        for label, micro_clusters in mcs_per_label.items():
            # merge clusters with the same label into a big cluster
            cluster = copy.copy(micro_clusters[0])
            for mc in range(1, len(micro_clusters)):
                cluster.merge(micro_clusters[mc])

            clusters[label] = cluster

        return len(clusters), clusters

    def _build(self):

        start_time_total = time.time()

        self.time_period_check()
        
        print("\n>> Timestamp: ", self.timestamp)
        print("> count_potential", len(self.p_micro_clusters))
        print("> count_outlier", len(self.o_micro_clusters))
        
        if len(self.p_micro_clusters) < self.m_minPoints:
            print("no building possible since num_potential_dbs < minPoints")
            return
        
        if self.save_partitions:
            df_partition              = self.remove_oldest_points_in_micro_clusters_timestamp()
            len_points                = df_partition.shape[0]
            len_mcs                   = len(self.p_micro_clusters)
            matrix_partitions         = [[-1 for j in range(len_mcs + 10)] for i in range(self.m_minPoints + 10)]
            matrix_partitions_mcs     = [[-1 for j in range(len_points + 10)] for i in range(self.m_minPoints + 10)]
            matrix_partitions_hdbscan = [[-1 for j in range(len_points + 10)] for i in range(self.m_minPoints + 10)]
            vertices                  = None
            
            self.micro_clusters_to_points(self.timestamp)
    
        start_hierarchies = time.time()
        
        # MinPts_Max > MinPts_Min
        for minpts in range(self.m_minPoints, 1, -self.step):
            print("\n-------------------------------------------------------------------------------------")
            print("MinPts: ", minpts)
            
            start_time_minpts = time.time()

            Vertex.s_idCounter = 0
        
            for mc in self.p_micro_clusters.values():
                mc.setVertexRepresentative(None)
                mc.setStaticCenter(self.timestamp)
            
            G = nx.Graph()
        
            start_mrg = time.time()
            mrg       = MutualReachabilityGraph(G, self.p_micro_clusters.values(), minpts, self.timestamp)
            mrg.buildGraph()
            end_mrg   = time.time()
            
            print("> Time for MRG: ", end_mrg - start_mrg)
            
            start_mst = time.time()
            T         = nx.minimum_spanning_tree(G)
            mst_max   = MinimalSpaningTree(T)
            mst_max.buildGraph()
            end_mst   = time.time()
            
            print("> Time for MST: ", end_mst - start_mst)
            
            self.m_update = Updating(mrg, mst_max)
            
            start_dendrogram = time.time()
            mst              = self.m_update.getMST()
            dendrogram       = Dendrogram(mst, minpts, self.timestamp)
            dendrogram.build()
            end_dendrogram   = time.time()
            
            end_time_minpts = time.time()
            print("> Total Time MinPts: ", end_time_minpts - start_time_minpts)
            start_selection = end_selection = 0
    
            # Time Selection CLusters
            if self.save_partitions:
                vertices          = mst
                start_selection   = time.time()
                selection         = dendrogram.clusterSelection()

                # Partitions HAStream MinPts
                cont          = 1
                partition_mcs = {}
            
                for n in selection:
                    it = iter(n.getVertices())

                    for el in it:
                        partition_mcs[el.getMicroCluster().getID()] = cont
                        matrix_partitions[minpts][el.getID()]               = cont

                    cont += 1

                cont = 0
                for i, row in df_partition.iterrows():
                    if row['id_mc'] in partition_mcs:
                        matrix_partitions_mcs[minpts][cont] = partition_mcs[row['id_mc']]
                    cont += 1
            
                end_selection = time.time()
                #print("> Time for Selection Clusters: ", end_selection - start_selection)
            
            # Partitions HDBSCAN
            if self.save_partitions:
                start_hdbscan = time.time()
                
                clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=minpts, match_reference_implementation = True)
                clusterer.fit(df_partition.drop("id_mc", axis=1))
                labels = clusterer.labels_

                p = 0
                for i in labels:
                    matrix_partitions_hdbscan[minpts][p] = i
                    p += 1
                    
                end_hdbscan = time.time()

                # Plot HDBSCAN result
                if self.plot:
                    self.plot_hdbscan_result(minpts, labels, df_partition)

            end_time_minpts = time.time()
            
            if self.runtime:
                self.df_runtime.at[minpts, 'minpts']         = minpts
                self.df_runtime.at[minpts, 'micro_clusters'] = len(self.p_micro_clusters)
                self.df_runtime.at[minpts, 'mrg']            = (end_mrg - start_mrg)
                self.df_runtime.at[minpts, 'mst']            = (end_mst - start_mst)
                self.df_runtime.at[minpts, 'dendrogram']     = (end_dendrogram - start_dendrogram)
                self.df_runtime.at[minpts, 'selection']      = (end_selection - start_selection) 
                self.df_runtime.at[minpts, 'total']          = (end_time_minpts - start_time_minpts)
            
            G          = None
            mrg        = None
            mst        = None
            mst_max    = None
            dendrogram = None
            selection  = None
            clusterer  = None
        
        # Time final timestamp
        if self.runtime:
            self.df_runtime_final.at[self.timestamp, 'timestamp']            = self.timestamp
            self.df_runtime_final.at[self.timestamp, 'micro_clusters']       = len(self.p_micro_clusters)
            self.df_runtime_final.at[self.timestamp, 'multiple_hierarchies'] = time.time() - start_hierarchies

        print(">Time Total: ", time.time() - start_time_total)
        
        if self.save_partitions:
            self.save_partitions_final(matrix_partitions, len_mcs, vertices.getVertices(), self.m_minPoints)
            self.save_partitions_mcs_and_points_minpts(len_points, matrix_partitions_mcs, matrix_partitions_hdbscan, 2, self.m_minPoints)
        
        if self.plot:
            self.plot_partitions(matrix_partitions, len_mcs, self.m_minPoints, df_partition)
        # reset df_mc_to_points['id_mc'] in timestamp
        # df_mc_to_points['id_mc'] = -1
        
    def _initial_single_linkage(self):
        start = time.time()
        
        self._init_buffer = np.array(self._init_buffer)
        
        # The linkage="single" does a clustering, e. g., the clusters are indentified and form big data bubbles.
        clustering = AgglomerativeClustering(n_clusters = int(self.n_samples_init*self.percent), linkage='average')
        clustering.fit(self._init_buffer)
        
        labels          = clustering.labels_
        labels_visited  = np.zeros(len(labels))
        len_buffer      = len(self._init_buffer)
        count_potential = 0
        min_mc = max_mc = 0
        epsilon         = {}
        pos_point       = 0
        
        for i in range(len_buffer):
            
            labels_visited[labels[i]] += 1
            
            if labels_visited[labels[i]] == 1:
                mc = MicroCluster(
                    x = dict(zip([j for j in range(len(self._init_buffer[i]))], self._init_buffer[i])),
                    timestamp=self.timestamp,
                    decaying_factor=self.decaying_factor,
                )

                mc.setID(labels[i])
                
                self.p_micro_clusters.update({labels[i]: mc})
                
            else:
                self.p_micro_clusters[labels[i]].insert(dict(zip([j for j in range(len(self._init_buffer[i]))], self._init_buffer[i])), self.timestamp)

                if labels_visited[labels[i]] == self.mu:
                    count_potential += 1
                if self.p_micro_clusters[labels[i]].getN() >= self.mu:
                    epsilon[labels[i]] = self.p_micro_clusters[labels[i]].getRadius(self.timestamp)
                
            max_mc = max(max_mc, self.p_micro_clusters[labels[i]].getN())
        
            df_mc_to_points.at[pos_point, 'id_mc'] = labels[i]
            pos_point += 1
        
        # outliers data_bubbles
        if count_potential != len(self.p_micro_clusters):
            key   = 0
            key_p = 0
            key_o = 2
            
            while labels_visited[key]:
                if labels_visited[key] < self.mu:
                    df_mc_to_points['id_mc'] = df_mc_to_points['id_mc'].replace(key, (-1) * key_o)
                    
                    self.o_micro_clusters[key_o] = self.p_micro_clusters[key]
                    self.p_micro_clusters.pop(key)

                    key_o += 1
                else:
                    if key != key_p:
                        self.p_micro_clusters[key].setID(key_p)
                        self.p_micro_clusters[key_p] = self.p_micro_clusters.pop(key)

                        if min_mc == 0:
                            min_mc = self.p_micro_clusters[key_p].getN()
                        else:
                            min_mc = min(min_mc, self.p_micro_clusters[key_p].getN())
                        
                        #update new key
                        df_mc_to_points['id_mc'] = df_mc_to_points['id_mc'].replace(key, key_p)

                    key_p += 1
                
                key += 1
                
        end = time.time()
        
        # Time
        if self.runtime:
            self.df_runtime_final.at[self.timestamp, 'summarization'] = end - start
        
        e_min  = min(epsilon.values())
        e_mean = sum(epsilon.values()) / count_potential
        e_max  = max(epsilon.values())
        
        self.epsilon = e_max
        
        print("> Total: ", (count_potential + len(self.o_micro_clusters)))
        print("> MCs Potential: ", count_potential)
        print("> Min_MC: ", min_mc)
        print("> Max_MC: ", max_mc)
        print("> Time for MCs: ", end - start)
        print("> Epsilon min: ", e_min)
        print("> Epsilon mean: ", e_mean)
        print("> Epsilon max: ", e_max)   
        
    def _expand_cluster(self, mc, neighborhood, id):
        for idx in neighborhood:
            item = self._init_buffer[idx]
            
            if not item.covered:
                item.covered = True
                mc.add(item.x)

                p = df_mc_to_points[(df_mc_to_points['x'] == item.x[0]) & (df_mc_to_points['y'] == item.x[1])]
                df_mc_to_points.at[p.index[0], 'id_mc'] = id

    def _get_neighborhood_ids(self, item):
        neighborhood_ids = deque()
        
        for idx, other in enumerate(self._init_buffer):
            if not other.covered:
                if self._distance(item.x, other.x) < self.epsilon:
                    neighborhood_ids.append(idx)
                    
        return neighborhood_ids
    
    def _initial_epsilon(self):
        start = time.time()
        
        for item in self._init_buffer:
            if not item.covered:
                item.covered = True
                neighborhood = self._get_neighborhood_ids(item)
                
                if len(neighborhood) > self.mu:
                    mc = MicroCluster(
                        x = item.x,
                        timestamp = self.timestamp,
                        decaying_factor = self.decaying_factor,
                    )
                    
                    id = len(self.p_micro_clusters)
                    
                    p = df_mc_to_points[(df_mc_to_points['x'] == item.x[0]) & (df_mc_to_points['y'] == item.x[1])]
                    df_mc_to_points.at[p.index[0], 'id_mc'] = id
                    
                    self._expand_cluster(mc, neighborhood, id)
                    mc.setStaticCenter(self.timestamp)
                    mc.setID(id)
                    
                    self.p_micro_clusters.update({id: mc})
                else:
                    item.covered = False

        # Outliers
        for item in self._init_buffer:
            if not item.covered:
                item.covered = True
                neighborhood = self._get_neighborhood_ids(item)
                
                mc = MicroCluster(x = item.x, timestamp = self.timestamp, decaying_factor = self.decaying_factor)

                id = 2 if len(self.o_micro_clusters) == 0 else list(self.o_micro_clusters.keys())[-1] + 1
                
                p = df_mc_to_points[(df_mc_to_points['x'] == item.x[0]) & (df_mc_to_points['y'] == item.x[1])]
                df_mc_to_points.at[p.index[0], 'id_mc'] = -id
                
                self._expand_cluster(mc, neighborhood, -id)
                mc.setStaticCenter(self.timestamp)
                mc.setID(id)
                
                self.o_micro_clusters.update({id: mc})
                
        end = time.time()
        
        # Time
        if self.runtime:
            self.df_runtime_final.at[self.timestamp, 'summarization'] = end - start
        
        print(">> Time Initial Summartization: ", end - start)
        
    def time_period_check(self):
        for i, p_micro_cluster_i in list(self.p_micro_clusters.items()):
            if p_micro_cluster_i.getWeight(self.timestamp) < self.mu * self.beta:
                # c_p became an outlier and should be deleted
                del self.p_micro_clusters[i]

        for j, o_micro_cluster_j in list(self.o_micro_clusters.items()):
            # calculate xi
            xi = (2**(-self.decaying_factor * (self.timestamp - o_micro_cluster_j.creation_time + self._time_period)) - 1) / (2 ** (-self.decaying_factor * self._time_period) - 1)                
            if o_micro_cluster_j.getWeight(self.timestamp) < xi:
                # c_o might not grow into a p-micro-cluster, we can safely delete it
                self.o_micro_clusters.pop(j)

    def learn_one(self, x, sample_weight=None):
        self._n_samples_seen += 1
        # control the stream speed
        
        if self._n_samples_seen % self.stream_speed == 0:
            self.timestamp += 1

        # Initialization
        if not self.initialized:
            if self.method_summarization == 'epsilon':
                self._init_buffer.append(self.BufferItem(x, self.timestamp, False))
            else:
                self._init_buffer.append(list(x.values()))
            
            if len(self._init_buffer) == self.n_samples_init:
                print("entrando no initial()")
                if self.method_summarization == 'epsilon':
                    self._initial_epsilon()
                else:
                    self._initial_single_linkage()
                
                print("-------------------")
                print("entrando no build()")
                self._build()
                
                if self.runtime:
                    self.save_runtime_timestamp()
                
                self.initialized = True
                del self._init_buffer
                
            return self

        # Merge
        self._merge(x)

        # Periodic cluster removal
        if self.timestamp > 0 and self.timestamp % self._time_period == 0:
            self.time_period_check()

        return self

    def predict_one(self, sample_weight=None):        
        # This function handles the case when a clustering request arrives.
        # implementation of the DBSCAN algorithm proposed by Ester et al.
        
        if not self.initialized:
            return 0
        
        self._build()
        
        if self.runtime:
            self.save_runtime_timestamp()
    
    def save_partitions_final(self,matrix_partitions, count_MC, vertices, minpts_max):
        m_directory = os.path.join(os.getcwd(), "results/flat_solutions")
        
        try:
            sub_dir = os.path.join(m_directory, "flat_solution_partitions_t" + str(self.timestamp))

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            if self.plot:
                cores = ["blue", "red", "orange", "green", "purple", "brown", "pink", "olive", "cyan"]
                
                for minpts in range(2, minpts_max + 1, self.step):
                    with open(os.path.join(sub_dir, "HAStream_Partitions_MinPts_" + str(minpts) + ".csv"), 'w') as writer:
                        writer.write("x,y,N,radio,color,cluster,ID\n")

                        for v in vertices:
                            if matrix_partitions[minpts][v.getID()] == -1:
                                writer.write(str(v.getMicroCluster().getCenter(self.timestamp)[0]) + "," + str(v.getMicroCluster().getCenter(self.timestamp)[1]) + "," + str(v.getMicroCluster().getWeight(self.timestamp)) + "," + str(v.getMicroCluster().getRadius(self.timestamp)) + ",black,-1," + str(v.getMicroCluster().getID()) + "\n")
                            else:
                                writer.write(str(v.getMicroCluster().getCenter(self.timestamp)[0]) + "," + str(v.getMicroCluster().getCenter(self.timestamp)[1]) + "," + str(v.getMicroCluster().getWeight(self.timestamp)) + "," + str(v.getMicroCluster().getRadius(self.timestamp)) + "," + cores[matrix_partitions[minpts][v.getID()] % 9] + "," + str(matrix_partitions[minpts][v.getID()]) + "," + str(v.getMicroCluster().getID()) + "\n")

            # Saving the partitions bubbles
            with open(os.path.join(sub_dir, "HAStream_All_Partitions_MinPts.csv"), 'w') as writer:
                for j in range(count_MC):
                    if j == 0:
                        writer.write(str(j))
                    else:
                        writer.write(", " + str(j))

                writer.write("\n")

                for i in range(2, minpts_max + 1, self.step):
                    for j in range(count_MC):
                        if j == 0:
                            writer.write(str(matrix_partitions[i][j]))
                        else:
                            writer.write(", " + str(matrix_partitions[i][j]))

                    writer.write("\n")
            
        except FileNotFoundError as e:
            print(e)
            
    def plot_partitions(self, matrix_partitions, count_DB, minpts_max, df_partition):
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        plot_kwds = {'s' : 2, 'linewidths':0}
        
        m_directory = os.path.join(os.getcwd(), "results/plots")
        
        try:
            sub_dir = os.path.join(m_directory, "plot_mcs_t" + str(self.timestamp))

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
                
            for minpts in range(2, minpts_max + 1, self.step):
                partition = pd.read_csv('results/flat_solutions/flat_solution_partitions_t' + str(self.timestamp) + '/HAStream_Partitions_MinPts_' + str(minpts) + '.csv', sep=',')

                # Statistic partition-------------------------------
                count_outlier = 0
                count_cluster = 0

                for j in range(len(partition)):

                    if(partition['cluster'].loc[j] == -1):
                        count_outlier += 1

                    if(partition['cluster'].loc[j] > count_cluster):
                        count_cluster = partition['cluster'].loc[j]

                legend  = ""
                legend += "MinPts: " + str(minpts) + "  "
                legend += "| Outliers: " + str(int((count_outlier * 100.0) / len(partition))) + "%  "
                legend += "| Clusters: " + str(count_cluster) + "  "
                legend += "| MCs: " + str(len(partition)) + "  "
                legend += "| Timestamp: " + str(self.timestamp)
                # -------------------------------------------------

                plt.figure(figsize = (16,12))

                for j in range(len(partition)):        
                    plt.gca().add_patch(plt.Circle((partition['x'].loc[j], partition['y'].loc[j]), partition['radio'].loc[j], color=partition['color'].loc[j], fill=False))

                #start = ((self._n_samples_seen/7000 ) - 1) * self.n_samples_init
                #end   = start + self.n_samples_init

                plt.scatter(df_partition[0], df_partition[1], c='black', **plot_kwds, label=legend)
                #plt.scatter(data[int(start):int(end), 0], data[int(start):int(end), 1], **plot_kwds, label=legend)
                #plt.scatter(data[0], data[1], **plot_kwds, label=legend)
                plt.legend(bbox_to_anchor=(-0.1, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, fontsize=28)

                plt.savefig("results/plots/plot_mcs_t" + str(self.timestamp) + "/minpts_" + str(minpts) + ".png")
                plt.close()
                
        except FileNotFoundError as e:
            print(e)

    def plot_hdbscan_result(self, minpts, labels, df_partition):
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        plot_kwds = {'s' : 10, 'linewidths':0}

        plt.figure(figsize = (16,12))
        title  = ""
        title += "HDBSCAN MinPts: " + str(minpts) + " | "
        title += "Clusters: " + str(len(set(labels)) - (1 if -1 in labels else 0)) + " | "
        title += "Outliers: " + str(np.sum(labels == -1)) + " | "
        title += "Len Points: " + str(len(labels))
        plt.title(title)

        plt.scatter(df_partition[0], df_partition[1], c=labels, cmap='magma', **plot_kwds)

        m_directory = os.path.join(os.getcwd(), "results/plots")
        sub_dir     = os.path.join(m_directory, "plot_mcs_t" + str(self.timestamp))

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        
        plt.savefig("results/plots/plot_mcs_t" + str(self.timestamp) + "/minpts_" + str(minpts) + "_hdbscan.png")
        
        plt.close()

    def save_partitions_mcs_and_points_minpts(self, len_partitions, matrix_partitions_mc, matrix_partitions_hdbscan, min_pts_min, min_pts_max):

        m_directory = os.path.join(os.getcwd(), "results/flat_solutions")
        
        try:
            sub_dir = os.path.join(m_directory, "flat_solution_partitions_t" + str(self.timestamp))

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            # Partitions HAStream MinPts
            with open(os.path.join(sub_dir, "all_partitions_mcs.csv"), 'w') as writer:
                for j in range(len_partitions):
                    if j == 0:
                        writer.write(str(j))
                    else:
                        writer.write(", " + str(j))

                writer.write("\n")

                for i in range(min_pts_min, min_pts_max + 1, self.step):
                    for j in range(len_partitions):
                        if j == 0:
                            writer.write(str(matrix_partitions_mc[i][j]))
                        else:
                            writer.write(", " + str(matrix_partitions_mc[i][j]))

                    writer.write("\n")

            # Partitions HDBSCAN MinPts
            with open(os.path.join(sub_dir, "all_partitions_hdbscan.csv"), 'w') as writer:
                for j in range(len_partitions):
                    if j == 0:
                        writer.write(str(j))
                    else:
                        writer.write(", " + str(j))

                writer.write("\n")

                for i in range(min_pts_min, min_pts_max + 1, self.step):
                    for j in range(len_partitions): 
                        if j == 0:
                            writer.write(str(matrix_partitions_hdbscan[i][j]))
                        else:
                            writer.write(", " + str(matrix_partitions_hdbscan[i][j]))

                    writer.write("\n")

        except FileNotFoundError as e:
            print(e)
    
    def save_runtime_timestamp(self):
        m_directory = os.path.join(os.getcwd(), "results/runtime")
        
        try:
            if not os.path.exists(m_directory):
                os.makedirs(m_directory)

            with open(os.path.join(m_directory, "runtime_t" + str(self.timestamp) + ".csv"), 'w') as writer:
                writer.write("minpts,mrg,mst,dendrogram,selection,total\n")

                for _, linha in self.df_runtime.iterrows():
                    writer.write(str(linha['minpts']) + ',' + str(linha['mrg']) + ',' + str(linha['mst']) + ',' + str(linha['dendrogram']) + ',' + str(linha['selection']) + ',' + str(linha['total']) + "\n")

        except FileNotFoundError as e:
            print(e)
            
    def save_runtime_final(self):
        m_directory = os.path.join(os.getcwd(), "results/runtime")
        
        try:
            if not os.path.exists(m_directory):
                os.makedirs(m_directory)

            with open(os.path.join(m_directory, "runtime_final_t" + str(self.timestamp) + ".csv"), 'w') as writer:
                writer.write("timestamp,micro_clusters,summarization,multiple_hierarchies\n")

                for _, linha in self.df_runtime_final.iterrows():
                    writer.write(str(linha['timestamp']) + ',' + str(linha['micro_clusters']) + ',' + str(linha['summarization']) + ',' + str(linha['multiple_hierarchies']) + "\n")

        except FileNotFoundError as e:
            print(e)
    
    def remove_oldest_points_in_micro_clusters_timestamp(self):
        # Remove oldest objects from removed MCs
        for i, row in df_mc_to_points.iterrows():
            if i <= self._n_samples_seen:
                if row['id_mc'] not in self.p_micro_clusters and row['id_mc'] > -1:
                    df_mc_to_points.at[i, 'id_mc'] = -1
                elif ((-1) * row['id_mc']) not in self.o_micro_clusters and row['id_mc'] < -1:
                    df_mc_to_points.at[i, 'id_mc'] = -1
        
        # Remove oldest objects inside of MCs
        max_id         = max(self.p_micro_clusters.keys())
        labels_visited = np.zeros(max_id + 1)
        diff_points    = np.zeros(max_id + 1)
        
        for i in range(self._n_samples_seen):
            id_mc = df_mc_to_points.loc[i, 'id_mc']
            
            if id_mc > -1:
                if not labels_visited[id_mc] and id_mc in self.p_micro_clusters:
                    labels_visited[id_mc] = df_mc_to_points[df_mc_to_points['id_mc'] == id_mc].shape[0]
                    n = self.p_micro_clusters[id_mc].getWeight(self.timestamp)
                    
                    diff_points[id_mc] = int(labels_visited[id_mc] - n)
                    
                if diff_points[id_mc]:
                    diff_points[id_mc] -= 1
                    df_mc_to_points.loc[i, 'id_mc'] = -1
                    
        del labels_visited
        del diff_points 
        
        return df_mc_to_points[df_mc_to_points['id_mc'] != -1]
    
    def micro_clusters_to_points(self, timestamp):

        m_directory = os.path.join(os.getcwd(), "results/datasets")
        
        try:
            if not os.path.exists(m_directory):
                os.makedirs(m_directory)
                        
            df_mc_to_points[(df_mc_to_points['id_mc'] != -1)].to_csv('results/datasets/data_t' + str(self.timestamp) + '.csv', index=False)

            if self.plot:
                sns.set_context('poster')
                sns.set_style('white')
                sns.set_color_codes()

                plot_kwds = {'s' : 1, 'linewidths':0}

                plt.figure(figsize=(12, 10))

                for key, value in self.p_micro_clusters.items():
                    plt.gca().add_patch(plt.Circle((value.getCenter(timestamp)[0], value.getCenter(timestamp)[1]), value.getRadius(timestamp), color='red', fill=False))

                for key, value in self.o_micro_clusters.items():
                    plt.gca().add_patch(plt.Circle((value.getCenter(timestamp)[0], value.getCenter(timestamp)[1]), value.getRadius(timestamp), color='blue', fill=False))

                df_plot = df_mc_to_points[(df_mc_to_points['id_mc'] != -1)]

                cmap = plt.get_cmap('tab10', len(list(set([row['id_mc'] for i, row in df_plot.iterrows()]))))

                plt.title("Timestamp: " + str(self.timestamp) + " | # Points: " + str(df_plot.shape[0]) + " | # MCs: " + str(len(self.p_micro_clusters)), fontsize=20)
                plt.scatter(df_plot[0], df_plot[1], c='green', **plot_kwds)
                plt.savefig("results/datasets/plot_dataset_t" + str(self.timestamp) + ".png")
                plt.close()
            
        except FileNotFoundError as e:
            print(e)

if __name__ == "__main__":

    data = pd.read_csv(sys.argv[1], sep=',')
    initial_points = int(sys.argv[2])
    
    scaler = MinMaxScaler()

    scaler.fit(data)
    
    data = pd.DataFrame(data=scaler.transform(data))
    
    # MC to points
    df_mc_to_points = data.copy()
    df_mc_to_points['id_mc'] = -1
    
    data = data.to_numpy()
    
    hdbstream = HDBStream(int(sys.argv[3]), 
                      step=2,
                      decaying_factor=float(sys.argv[4]),
                      beta = 0.75,
                      mu=2,
                      n_samples_init=initial_points,
                      epsilon = 0.005,
                      stream_speed=100,
                      percent=0.10,
                      method_summarization='single',
                      runtime=True,
                      plot=False,
                      save_partitions=False)

    count_points = 0
    
    for x, _ in stream.iter_array(data):
        _ = hdbstream.learn_one(x)
    
        count_points += 1
    
        if not (count_points % initial_points) and count_points != initial_points:
            hdbstream.predict_one()
    hdbstream.save_runtime_final()
