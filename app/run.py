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

from matplotlib import axes
from river import stream
from river import base, utils
from abc import ABCMeta
from collections import defaultdict, deque
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

class Vertex():

    s_idCounter = 0

    def __init__(self, db, timestamp, id=None):
        self.m_id          = id if id is not None else Vertex.s_idCounter
        Vertex.s_idCounter += 1
        self.m_db          = db
        self.timestamp     = timestamp
        
        if self.m_db is not None:
            self.m_db.setVertexRepresentative(self)
            
        self.m_visited            = False
        self.m_coreDistanceObject = None
        self.m_lrd                = -1
        self.m_coreDist           = 0
        
    def getDataBubble(self):
        return self.m_db

    def getCoreDistance(self):
        if self.m_coreDist == 0:
            return -1
        return self.m_coreDist

    def setCoreDist(self, coreDistValue):
        self.m_coreDist = coreDistValue

    def setCoreDistance(self, coreDistObj):
        self.m_coreDistanceObject = coreDistObj

    def String(self):
        return f"({self.m_db.getRep()})"

    def getGraphVizVertexString(self):
        return f"vertex{self.m_id}"

    def getGraphVizString(self):
        #return f"{self.getGraphVizVertexString()} [label = " {self.String}  cdist={self.getCoreDistance()}"];"
        return "{} [label=\"{}\"]" .format(self.getGraphVizVertexString(),self.getGraphVizVertexString())

    def getDistanceToVertex(self, other):
        return self.m_db.getCenterDistance(other.getDataBubble())
    
    def getDistanceRep(self, vertex):
        x1 = self.distance(self.m_db.getRep(self.timestamp), vertex.getDataBubble().getRep(self.timestamp))
        
        return x1

    def getDistance(self, vertex):
        if self.m_db.getStaticCenter() is None or vertex.getDataBubble().getStaticCenter() is None:
            return self.getDistanceToVertex(vertex)
        
        x1 = self.distance(self.m_db.getRep(self.timestamp), vertex.getDataBubble().getRep(self.timestamp)) - (self.m_db.getExtent(self.timestamp) + vertex.getDataBubble().getExtent(self.timestamp))
        x2 = self.m_db.getNnDist(1, self.timestamp)
        x3 = vertex.getDataBubble().getNnDist(1, self.timestamp)
        
        if x1 >= 0:
            return x1 + x2 + x3
        
        return max(x2, x3)
    
        #return self.distance(self.m_db.getRep(self.timestamp), vertex.getDataBubble().getRep(self.timestamp))

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

class DataBubble(metaclass=ABCMeta):
    
    s_idCounter = 0
  
    def __init__(self, x, timestamp, decaying_factor):

        self.x = x
        
        self.db_id              = DataBubble.s_idCounter
        DataBubble.s_idCounter += 1
        
        self.last_edit_time  = timestamp
        self.creation_time   = timestamp
        self.decaying_factor = decaying_factor

        self.N              = 1
        self.linear_sum     = x
        self.squared_sum    = {i: (x_val * x_val) for i, x_val in x.items()}        
        self.m_staticCenter = len(self.linear_sum)
    
    def getID(self):
        return self.db_id

    def setID(self, id):
        self.db_id = id
    
    def getN(self):
        return self.N

    def _weight(self, timestamp):
        return self.N * self.fading_function(timestamp - self.last_edit_time)
        
    def getRep(self, timestamp):
        ff     = self.fading_function(timestamp - self.last_edit_time)
        weight = self._weight(timestamp)
        center = {key: (val * ff) / weight for key, val in self.linear_sum.items()}
        
        return center

    def getExtentDB(self, timestamp):        
        x1  = 0
        x2  = 0
        res = 0
        
        ff     = self.fading_function(timestamp - self.last_edit_time)
        weight = self._weight(timestamp)
        
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

        return (res / len(self.linear_sum)) * 1.4 #redius factor
        #return res

    def getExtent(self, timestamp):        
        x1  = 0
        x2  = 0
        res = 0
        
        ff     = self.fading_function(timestamp - self.last_edit_time)
        weight = self._weight(timestamp)
    
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            val_ss = self.squared_sum[key]
            
            # raio Micro-Cluster
            x1  = (val_ss * ff) / weight
            x2  = ((val_ls * ff) / weight)**2
            tmp = (x1 - x2)
            
            res += math.sqrt(tmp) if tmp > 0 else (1/10 * 1/10)
            
        return (res / len(self.linear_sum)) * 1.8  #redius factor
        #return res

    def insert(self, x, timestamp):
        
        if self.last_edit_time != timestamp:
            self.fade(timestamp)
        
        self.last_edit_time = timestamp
        
        self.N += 1
        
        for key, val in x.items():
            try:
                self.linear_sum[key]  += val
                self.squared_sum[key] += val * val
            except KeyError:
                self.linear_sum[key]  = val
                self.squared_sum[key] = val * val
    
    def fade(self, timestamp):
        ff = self.fading_function(timestamp - self.last_edit_time)
        
        self.N *= ff
        
        for key, val in self.linear_sum.items():
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

    def getNnDist(self, k, timestamp):
        return ((k / self.N)**(1.0 / len(self.linear_sum))) * self.getExtent(timestamp)

    def fading_function(self, time):
        return 2 ** (-self.decaying_factor * time)
    
    def setVertexRepresentative(self, v : Vertex): 
        self.m_vertexRepresentative = v

    def getVertexRepresentative(self):
        return self.m_vertexRepresentative
    
    def getStaticCenter(self):
        return self.m_staticCenter

    def setStaticCenter(self, timestamp):
        m_static_center = self.getRep(timestamp).copy()
        return m_static_center

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
    
    def __str__(self):
        return str(self.m_weight)

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
        #return f"{self.m_vertex1.getGraphVizVertexString()} -- {self.m_vertex2.getGraphVizVertexString()}"
        return "{} -- {} [label=\"{}\"]".format(self.m_vertex1.getGraphVizVertexString(),self.m_vertex2.getGraphVizVertexString(),self.getWeight())

    def setEdgeWeight(self, weight):
        self.m_weight = weight

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

    def getGraphVizString(self, timestamp, minpts):
        
        m_directory = os.path.join(os.getcwd(), "results/graphviz")
        
        try:
            sub_dir = os.path.join(m_directory, "graphviz_t" + str(timestamp))

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            with open(os.path.join(sub_dir, "graphviz_MinPts_" + str(minpts) + ".txt"), 'w') as writer:
                writer.write("graph {\n")

                edges = set()

                vertices = sorted(self.m_graph, key=lambda x: x.getID())
                
                for v in vertices:
                    writer.write("\t" + v.getGraphVizString() + "\n")
                    edges.update(self.adjacencyList(v).values())

                edges_sorted = sorted(edges, key=lambda x: x.getWeight())

                for e in edges_sorted:
                    writer.write("\t" + e.graphVizString() + "\n")

                writer.write("}\n")

        except FileNotFoundError as e:
            print(e)

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
        
class MutualReachabilityGraph(AbstractGraph):
    def __init__(self, G, dbs : DataBubble, minPts, timestamp):
        super().__init__()
        self.m_minPts  = minPts
        self.G         = G
        self.timestamp = timestamp
        self.ids       = []
        
        # plot DB core distance
        self.id_db     = 344
        self.neighbour = 0

        for db in dbs:
            v = Vertex(db, timestamp)
            db.setVertexRepresentative(v)
            self.G.add_node(v)
            
            self.addVertex(v)

        self.knng = KNearestNeighborsGraph(G)
        
        start = time.time()
        self.computeCoreDistance(G, minPts)
        end   = time.time()
        print("> Time coreDistanceDB", end - start, end='\n')

    def getKnngGraph(self):
        return self.knng
    
    def buildAbsGraph(self):
        
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        plot_kwds = {'s' : 1, 'linewidths':0}
        
        plt.figure(figsize = (16,12))
        
        linhas = []
        
        for i, (u,v,w) in enumerate(self.G.edges(data='weight')):
            self.addVertex(u)
            self.addVertex(v)
            self.addEdge(u,v,w)
            
            plt.gca().add_patch(plt.Circle((u.getDataBubble().getRep(self.timestamp)[0],u.getDataBubble().getRep(self.timestamp)[1]), u.getDataBubble().getExtent(self.timestamp), color='blue', fill=False))
            plt.gca().add_patch(plt.Circle((v.getDataBubble().getRep(self.timestamp)[0],v.getDataBubble().getRep(self.timestamp)[1]), v.getDataBubble().getExtent(self.timestamp), color='blue', fill=False))
            
            linhas.append(((u.getDataBubble().getRep(self.timestamp)[0],u.getDataBubble().getRep(self.timestamp)[1]) ,(v.getDataBubble().getRep(self.timestamp)[0],v.getDataBubble().getRep(self.timestamp)[1]), w))
            
            plt.text(u.getDataBubble().getRep(self.timestamp)[0], u.getDataBubble().getRep(self.timestamp)[1], str(u.getID()), fontsize=18, ha='center', va='center')
            plt.text(v.getDataBubble().getRep(self.timestamp)[0], v.getDataBubble().getRep(self.timestamp)[1], str(v.getID()), fontsize=18, ha='center', va='center')
        
        # Loop através da lista de linhas
        for (x1, y1), (x2, y2), numero in linhas:
            # Trace a linha
            plt.plot([x1, x2], [y1, y2], marker='o', linestyle='-', markersize=5, label=str(numero))

            # Adicione o número como texto no meio da linha
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(numero), fontsize=12, ha='center', va='center')

        # Mostre o gráfico
        plt.scatter(data[0:5000, 0], data[0:5000, 1], **plot_kwds)
        plt.show()
       
    def buildGraph(self):
        for v1 in self.G:            
            for v2 in self.G:
                if v1 != v2:
                    mrd = self.getMutualReachabilityDistance(v1, v2)
                    self.G.add_edge(v1, v2, weight = mrd)
                    self.addEdge(v1, v2, mrd)

    def computeCoreDistance(self, vertices, minPts):
        for current in vertices:
            if current.getDataBubble()._weight(self.timestamp) >= minPts:
                current.setCoreDist(current.getDataBubble().getNnDist(minPts, self.timestamp))
            else:
                neighbours  = self.getNeighbourhood(current, vertices)
                countPoints = current.getDataBubble()._weight(self.timestamp)
                neighbourC  = None
                
                #print("<<< Vertice", current.getID())

                for n in neighbours:
                    weight      = n.getVertex().getDataBubble()._weight(self.timestamp)
                    countPoints += weight
                    
                    #print("<<<<<<< Vertice", n.getVertex().getID())
                    
                    if current.getID() == self.id_db:
                        #print('Dist neghbour: ', n.getDistance())
                        #print('ID: ', n.getVertex().getID())
                        self.ids.append(n.getVertex().getID())
                    
                    if self.knng.getEdge(current, n.getVertex()) is None:
                        self.knng.setEdge(current, n.getVertex())
                        
                    if countPoints >= minPts:
                        countPoints   -= weight
                        neighbourC     = n
                        
                        if current.getID() == self.id_db:
                            self.neighbour = n.getVertex().getID()
                        
                        break
                
                extentCurrent    = current.getDataBubble().getExtent(self.timestamp)
                extentNeighbourC = neighbourC.getVertex().getDataBubble().getExtent(self.timestamp)
                
                overlapping = current.getDistanceRep(neighbourC.getVertex()) - (extentCurrent + extentNeighbourC)
                
                knnDistNeighbourC = neighbourC.getVertex().getDataBubble().getNnDist(minPts - countPoints, self.timestamp)
                
                if(overlapping >= 0.0):
                    current.setCoreDist(current.getDistanceRep(neighbourC.getVertex()) - extentNeighbourC + knnDistNeighbourC)
                else:
                    overlapping *= -1
                    
                    if knnDistNeighbourC <= overlapping:
                        current.setCoreDist(current.getDataBubble().getExtent(self.timestamp))
                    else:
                        current.setCoreDist(current.getDataBubble().getExtent(self.timestamp) + knnDistNeighbourC - overlapping)
                
    def getNeighbourhood(self, vertex, vertices):
        neighbours = []
        
        for v in vertices:
            if v != vertex:
                neighbour = Neighbour(v, vertex.getDistanceRep(v) - v.getDataBubble().getExtent(self.timestamp))
                neighbours.append(neighbour)
                
        neighbours.sort(key=lambda x: x.getDistance(), reverse=False)
        
        return neighbours

    def getMutualReachabilityDistance(self, v1, v2):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))

class MinimalSpaningTree(AbstractGraph):
    def __init__(self, graph):
        super().__init__()
        self.m_inputGraph = graph
        #self.mst_to_hdbscan = []

    def buildGraph(self):
        for i, (u,v,w) in enumerate(self.m_inputGraph.edges(data='weight')):
            self.addVertex(u)
            self.addVertex(v)
            self.addEdge(u,v,w)
        #    self.mst_to_hdbscan.append([u.getID() , v.getID() , w]) 
        #print(self.mst_to_hdbscan)
    
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

    def buildAbsGraph(self, timestamp):
        
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        plot_kwds = {'s' : 1, 'linewidths':0}
        
        plt.figure(figsize = (16,12))
        
        linhas = []
        
        for edge in self.getEdges():
            u = edge.getVertex1()
            v = edge.getVertex2()

            plt.gca().add_patch(plt.Circle((u.getDataBubble().getRep(timestamp)[0],u.getDataBubble().getRep(timestamp)[1]), u.getDataBubble().getExtent(timestamp), color='blue', fill=False))
            plt.gca().add_patch(plt.Circle((v.getDataBubble().getRep(timestamp)[0],v.getDataBubble().getRep(timestamp)[1]), v.getDataBubble().getExtent(timestamp), color='blue', fill=False))
            
            linhas.append(((u.getDataBubble().getRep(timestamp)[0],u.getDataBubble().getRep(timestamp)[1]) ,(v.getDataBubble().getRep(timestamp)[0],v.getDataBubble().getRep(timestamp)[1])))
            
            plt.text(u.getDataBubble().getRep(timestamp)[0], u.getDataBubble().getRep(timestamp)[1], str(u.getID()), fontsize=10, ha='center', va='center')
            plt.text(v.getDataBubble().getRep(timestamp)[0], v.getDataBubble().getRep(timestamp)[1], str(v.getID()), fontsize=10, ha='center', va='center')
        
        # Loop através da lista de linhas
        for (x1, y1), (x2, y2) in linhas:
            # Trace a linha
            plt.plot([x1, x2], [y1, y2], marker='o', linestyle='-', markersize=5)

        # Mostre o gráfico
        plt.scatter(data[0:5000, 0], data[0:5000, 1], **plot_kwds)
        plt.show()
    
class KNearestNeighborsGraph(AbstractGraph):
    def __init__(self, vertices):
        super().__init__()
        
        for v in vertices:
            super().addVertex(v)
        
    def setEdge(self, v1, v2):
        distance = v1.getDistance(v2)
        super().addEdge(v1, v2, distance)

    def buildGraph(self):
        pass

    def buildAbsGraph(self, timestamp):
        
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        plot_kwds = {'s' : 1, 'linewidths':0}
        
        plt.figure(figsize = (16,12))
        
        linhas = []
        
        for edge in self.getEdges():
            u = edge.getVertex1()
            v = edge.getVertex2()
            
            plt.gca().add_patch(plt.Circle((u.getDataBubble().getRep(timestamp)[0],u.getDataBubble().getRep(timestamp)[1]), u.getDataBubble().getExtent(timestamp), color='blue', fill=False))
            plt.gca().add_patch(plt.Circle((v.getDataBubble().getRep(timestamp)[0],v.getDataBubble().getRep(timestamp)[1]), v.getDataBubble().getExtent(timestamp), color='blue', fill=False))
            
            linhas.append(((u.getDataBubble().getRep(timestamp)[0],u.getDataBubble().getRep(timestamp)[1]) ,(v.getDataBubble().getRep(timestamp)[0],v.getDataBubble().getRep(timestamp)[1])))
            
            plt.text(u.getDataBubble().getRep(timestamp)[0], u.getDataBubble().getRep(timestamp)[1], str(u.getID()), fontsize=10, ha='center', va='center')
            plt.text(v.getDataBubble().getRep(timestamp)[0], v.getDataBubble().getRep(timestamp)[1], str(v.getID()), fontsize=10, ha='center', va='center')
        
        # Loop através da lista de linhas
        for (x1, y1), (x2, y2) in linhas:
            # Trace a linha
            plt.plot([x1, x2], [y1, y2], marker='o', linestyle='-', markersize=5)

        # Mostre o gráfico
        plt.scatter(data[0:5000, 0], data[0:5000, 1], **plot_kwds)
        plt.show()

class CoreSG(AbstractGraph):
    def __init__(self, mst: MinimalSpaningTree, knng: KNearestNeighborsGraph, timestamp):
        super().__init__()
        
        self.timestamp = timestamp
        
        for v in mst.getVertices():
            super().addVertex(v)
            
        self.addKnng(knng)
        self.addMst(mst)

    def buildGraph(self):
        pass
    
    def getGraphNetworkx(self):
        G = nx.Graph()
        
        for e in self.getEdges():
            v1 = e.getVertex1()
            v2 = e.getVertex2()
            G.add_node(v1)
            G.add_node(v2)
            G.add_edge(v1, v2, weight = e.getWeight())
            
        return G
    
    def addKnng(self, knng: KNearestNeighborsGraph):
        edges = knng.getEdges()  
        
        for e in edges:
            self.addEdge(e.getVertex1(), e.getVertex2(), e.getWeight())

    def addMst(self, mst: MinimalSpaningTree):
        for e in mst.getEdges():
            if self.getEdge(e.getVertex1(), e.getVertex2()) is None:
                self.addEdge(e.getVertex1(), e.getVertex2(), e.getWeight())

    def computeHierarchyMinPts(self, minPts: int):
        self.computeCoreDistance(minPts)
        edgesGraph = self.getEdges()
        
        for e in edgesGraph:
            self.removeEdge2(e)
            self.addEdge(e.getVertex1(), e.getVertex2(), self.getMutualReachabilityDistance(e.getVertex1(), e.getVertex2()))

    def computeCoreDistance(self, minPts: int):
        vertices = self.getVertices()
        
        for current in vertices:
            if current.getDataBubble()._weight(self.timestamp) >= minPts:
                nnDist = current.getDataBubble().getNnDist(minPts, self.timestamp)
                current.setCoreDist(nnDist)
            else:
                neighbours  = self.getNeighbourhoodMinPtsNN(current)
                countPoints = current.getDataBubble()._weight(self.timestamp)
                neighbourC  = None

                for n in neighbours:
                    weight      = n.getVertex().getDataBubble()._weight(self.timestamp)
                    countPoints += weight
                        
                    if countPoints >= minPts:
                        countPoints -= weight
                        neighbourC   = n
                        break
                
                extentCurrent    = current.getDataBubble().getExtent(self.timestamp)
                extentNeighbourC = neighbourC.getVertex().getDataBubble().getExtent(self.timestamp)
                
                overlapping = current.getDistanceRep(neighbourC.getVertex()) - (extentCurrent + extentNeighbourC)
                
                knnDistNeighbourC = neighbourC.getVertex().getDataBubble().getNnDist(minPts - countPoints, self.timestamp)
                
                if(overlapping >= 0.0):
                    current.setCoreDist(current.getDistanceRep(neighbourC.getVertex()) - extentNeighbourC + knnDistNeighbourC)
                else:
                    overlapping *= -1
                    
                    if knnDistNeighbourC <= overlapping:
                        current.setCoreDist(current.getDataBubble().getExtent(self.timestamp))
                    else:
                        current.setCoreDist(current.getDataBubble().getExtent(self.timestamp) + knnDistNeighbourC - overlapping)
                
    def getNeighbourhoodMinPtsNN(self, vertex):
        neighbours = []
        vertices   = self.getAdjacentEdges(vertex).keys()
        
        for v in vertices:
            if v != vertex:
                neighbour = Neighbour(v, vertex.getDistanceRep(v) - v.getDataBubble().getExtent(self.timestamp))
                neighbours.append(neighbour)
        
        neighbours.sort(key=lambda x: x.getDistance(), reverse=False)
        
        return neighbours

    def getMutualReachabilityDistance(self, v1: Vertex, v2: Vertex):
        return max(v1.getCoreDistance(), max(v2.getCoreDistance(), v1.getDistance(v2)))

class Updating:
    def __init__(self, mrg: MutualReachabilityGraph, mst : MinimalSpaningTree, csg : CoreSG):
        self.m_mrg = mrg
        self.m_mst = mst
        self.m_csg = csg
        self.m_globalReplacementEdge = None
    
    def getMST(self):
        return self.m_mst
    
    def getMRG(self):
        return self.m_mrg
    
    def getCSG(self):
        return self.m_csg

class Component(AbstractGraph):
    def __init__(self, startVertex: Vertex, graph: AbstractGraph, prepareEdges : bool):
        super().__init__()
        
        self.m_edges_summarized_by_weight = {}
        self.m_prepare_edges              = prepareEdges
        
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

    def __init__(self, c, timestamp):
        self.m_vertices   = set(c)
        self.m_children   = []
        self.m_delta      = True
        self.m_label      = Node.s_label
        self.m_id         = Node.s_label
        Node.s_label      += 1
        self.m_parent     = None
        self.m_scaleValue = 0
        self.m_stability  = 0.0
        self.timestamp    = timestamp
        
    def getID(self):
        return self.m_id
    
    def getInternalPoints(self):
        numPoints = 0.0

        for v in self.getVertices():
            numPoints += v.getDataBubble()._weight(self.timestamp)
            
        return numPoints
    
    def computeStabilityNew(self) -> float:
        self.m_stability = 0.0
        eps_max          = self.m_scaleValue
        
        for child in self.getChildren():
            self.m_stability += child.getInternalPoints() * ((1.0 / child.getScaleValue()) - (1.0 / eps_max))

        #print("<<< ", self.m_stability)
        return self.m_stability
    
    def computeStability(self) -> float:
        if self.m_parent is None:
            return float('nan')

        eps_max = self.m_parent.m_scaleValue
        eps_min = self.m_scaleValue
        
        # É o somatório dos pesos vezes a densidade minima (quando o Cluster foi criado) + a densidade máxima (quando o DB saiu do cluster)
        self.m_stability = len(self.m_vertices) * ((1 / eps_min) - (1 / eps_max))

        return self.m_stability

    def addChild(self, child: "Node"):
        self.m_children.append(child)

    def getChildren(self):
        return self.m_children
    
    def getChildrenMinClusterSize(self, min_cluster_size):
        children = []
        
        for child in self.getChildren():
            if child.getInternalPoints() >= min_cluster_size:
                children.append(child)
            
        return children

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
        return "node" + str(self.m_label)

    def getGraphVizEdgeLabelString(self):
        return "[label=\"{:.2f}\"];".format(self.m_scaleValue)

    def getGraphVizString(self):
        return "{} [label=\"Num={}[SV,SC,D,P]:{{ {:.4f}; {:.10f}; {}; ".format(self.getGraphVizNodeString(), len(self.m_vertices), self.m_scaleValue, self.m_stability, self.m_delta)

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
    def __init__(self, mst: MinimalSpaningTree, min_cluster_size: int, minPts: int,timestamp):        
        assert len(mst.getVertices()) > 0
        Node.resetStaticLabelCounter()

        self.m_components     = []
        self.m_minClusterSize = min_cluster_size
        self.m_minPts         = minPts
        first                 = None
        it                    = iter(mst.getVertices())
        self.timestamp        = timestamp
        
        if next(it):
            first = next(it)

        assert first is not None

        self.m_mstCopy = DendrogramComponent(first, mst, True)
        
        self.m_root = Node(self.m_mstCopy.getVertices(), self.timestamp)
        
        self.spurious_1   = 0
        self.spurious_gr2 = 0
        
        self.m_mstCopy.setNodeRepresentitive(self.m_root)
        self.m_components.append(self.m_mstCopy)
        self.len_mst = len(mst.getVertices())

    def build(self):
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
                        numPoints += v.getDataBubble()._weight(self.timestamp)
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
                        child = Node(c.getVertices(), self.timestamp)

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
    
    def getLeaves(self, node):
        res   = []
        queue = [node]

        while queue:
            n = queue.pop(0)
            
            count = 0
            
            for child in n.getChildrenMinClusterSize(self.m_minClusterSize):
                count += 1
                queue.append(child)
            
            if count == 0:
                res.append(n)

        return res

    # Compare to Nodes
    def compareNode(self, n: Node):
        return (n.getScaleValue(), n.getInternalPoints())
    
    def clusterSelection(self):
        selection = []

        # Step 1
        leaves = self.getLeaves(self.m_root)
        
        for leaf in leaves:
            #print("leaf ", leaf.getInternalPoints())
            leaf.setPropagatedStability(leaf.computeStability())
            #print("SC: ", leaf.getStability())

        # Special case
        if len(leaves) == 1 and leaves[0] == self.m_root:
            selection.append(self.m_root)
            
            return selection

        queue = []
        
        # add the Parent of the leaves
        for leaf in leaves:
            if leaf.getParent() is not None and leaf.getParent() not in queue:
                queue.append(leaf.getParent())
        
        queue.sort(key=self.compareNode)
        
        #for i in queue:
        #    print("ord: ", i.getInternalPoints())
        
        # Step 2
        while queue:
            current           = queue[0]
            current_stability = current.computeStability()
            
            children_sum_stability = 0.0
            
            #print("Pai ", current.getInternalPoints())
            
            #for child in current.getChildrenMinClusterSize(self.m_minClusterSize):
                #print("Child ", child.getInternalPoints())
            #    children_sum_stability += child.getPropagatedStability()
                
            children_sum_stability = sum(child.getPropagatedStability() for child in current.getChildrenMinClusterSize(self.m_minClusterSize))
            
            #print("> Stability Pai: ", current_stability)
            #print("> Stability Children: ", children_sum_stability)
            
            if current_stability < children_sum_stability:
                current.setPropagatedStability(children_sum_stability)
                current.resetDelta()
            else:
                current.setPropagatedStability(current_stability)
            
            for c in current.getChildrenMinClusterSize(self.m_minClusterSize):
                if c.getPropagatedStability() == 0:
                    c.resetDelta()
            
            queue.remove(current)
            
            if current.getParent() not in queue and current.getParent() is not None:
                queue.append(current.getParent())
        
            queue.sort(key=self.compareNode)

        # get clustering selection
        selection_queue = self.m_root.getChildrenMinClusterSize(self.m_minClusterSize).copy()
        self.m_root.resetDelta()

        while selection_queue:
            current = selection_queue.pop(0)
            
            if not current.isDiscarded():
                #print("Node: ", current.getInternalPoints())
                selection.append(current)
            else:
                selection_queue.extend(current.getChildrenMinClusterSize(self.m_minClusterSize))
                
        #self.condensedTreePlot(selection)
        return selection
    
    def getLeavesDfs(self, node):
        res = []
        
        if len(node.getChildrenMinClusterSize(self.m_minClusterSize)) == 0:
            res.append(node)
            return res

        for n in node.getChildrenMinClusterSize(self.m_minClusterSize):
            res.extend(self.getLeavesDfs(n))

        return res      
        
    def condensedTreePlot(self, selection, select_clusters = True, selection_palette = None, label_clusters = False):
        
        cluster_x_coords = {}

        leaves = self.getLeavesDfs(self.m_root)
        leaf_position = 0.0

        # set coords X from leaves
        for leaf in leaves:
            cluster_x_coords[leaf] = leaf_position
            leaf_position += 1
        
        # add the x and y coordinates for the clusters
        queue = []
        
        queue.extend(leaves)
        
        queue.sort(key = self.compareNode)
        
        cluster_y_coords = {self.m_root: 0.0}

        while queue:
            n        = queue[0]
            children = n.getChildrenMinClusterSize(self.m_minClusterSize)
            
            if len(children) > 1:
                left_child = children[0]
                right_child = children[1]
                
                mean_coords_children = (cluster_x_coords[left_child] + cluster_x_coords[right_child]) / 2.0
                cluster_x_coords[n] = mean_coords_children
                
                cluster_y_coords[left_child] = 1.0 / left_child.getScaleValue()
                cluster_y_coords[right_child] = 1.0 / right_child.getScaleValue()
                
            if n.getParent() is not None and n.getParent() not in queue:
                queue.append(n.getParent())

            queue.remove(n)
            queue.sort(key = self.compareNode)

        #print("> 1º While")

        # set scaling to plot
        root    = self.m_root
        scaling = 0
        
        for c in self.m_root.getChildren():
            scaling += len(c.getVertices())
        
        cluster_bounds = {}

        bar_centers = []
        bar_heights = []
        bar_bottoms = []
        bar_widths  = []

        # set bar configuration
        queue.clear()
        
        queue = [self.m_root]

        while queue:
            c = queue[0]
            
            cluster_bounds[c] = [0, 0, 0, 0]
            
            n_children = c.getChildren()
            
            if len(n_children) == 0:
                queue.remove(c)
                continue            
            
            current_size = 0           
            
            max_lambda = []
            for a in n_children:
                current_size += len(a.getVertices())
                max_lambda.append(1.0 / a.getScaleValue())
            
            current_lambda   = cluster_y_coords[c]
            cluster_max_size = current_size
            
            cluster_max_lambda = max_lambda[-1]
            
            cluster_min_size = 0
            
            for b in n_children:
                if (1.0 / b.getScaleValue())  == cluster_max_lambda:
                    cluster_min_size += len(b.getVertices())
            
            #2000
            max_rectangle_per_icicle = 20
            total_size_change        = float(cluster_max_size - cluster_min_size)
            step_size_change         = total_size_change / max_rectangle_per_icicle
            
            cluster_bounds[c][0] = cluster_x_coords[c] * scaling - (current_size / 2.0)
            cluster_bounds[c][1] = cluster_x_coords[c] * scaling + (current_size / 2.0)
            cluster_bounds[c][2] = cluster_y_coords[c]
            cluster_bounds[c][3] = cluster_max_lambda
            
            last_step_size   = current_size
            last_step_lambda = current_lambda
            
            
            for i in n_children:
                
                if (1.0 / i.getScaleValue())  != current_lambda and (last_step_size - current_size > step_size_change or (1.0 / i.getScaleValue()) == cluster_max_lambda):
                    bar_centers.append(cluster_x_coords[c] * scaling)
                    bar_heights.append((1.0 / i.getScaleValue()) - last_step_lambda)
                    bar_bottoms.append(last_step_lambda)
                    bar_widths.append(last_step_size)
                    last_step_size   = current_size
                    last_step_lambda = current_lambda
                else:
                    current_size -= len(i.getVertices())
                    
                current_lambda = 1.0 / i.getScaleValue()
            
            if c.getChildrenMinClusterSize(self.m_minClusterSize) is not None:
                queue.extend(c.getChildrenMinClusterSize(self.m_minClusterSize))

            queue.remove(c)

        #print("> 2º While")

        # set lines to plot
        line_xs = []
        line_ys = []

        queue_dendrogram = []
        queue_dendrogram.append(self.m_root)

        while queue_dendrogram:
            n = queue_dendrogram[0]
            children = n.getChildrenMinClusterSize(self.m_minClusterSize)

            for n_child in children:
                sign = 1

                if (cluster_x_coords[n_child] - cluster_x_coords[n]) < 0:
                    sign = -1

                line_xs.append((cluster_x_coords[n] * scaling, cluster_x_coords[n_child] * scaling + sign * (len(n_child.getVertices()) / 2.0)))
                line_ys.append((cluster_y_coords[n_child],cluster_y_coords[n_child]))
            
            if len(children) != 0:
                queue_dendrogram.extend(children)
                
            queue_dendrogram.remove(n)

        #print("> 3º While")
        
        fig, ax = plt.subplots(figsize=(16, 10))

        # Bars max(bar_widths)
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, self.len_mst))
        sm.set_array([x  for x in bar_widths ])
        bar_colors = [sm.to_rgba(x) for x in bar_widths]
        
        ax.bar(
            bar_centers,
            bar_heights,
            bottom=bar_bottoms,
            width=bar_widths,
            color=bar_colors,
            align='center',
            linewidth=0
        )
        
        for i in range(len(line_xs)):
            ax.plot(*[[line_xs[i][0], line_xs[i][1]], [line_ys[i][0], line_ys[i][1]]], color='black', linewidth=1)
        
        cluster_bounds2 = cluster_bounds
        
        if select_clusters:
            try:
                from matplotlib.patches import Ellipse
            except ImportError:
                raise ImportError('You must have matplotlib.patches available to plot selected clusters.')

            chosen_clusters = selection
            
            # Extract the chosen cluster bounds. If enough duplicate data points exist in the
            # data the lambda value might be infinite. This breaks labeling and highlighting
            # the chosen clusters.
            cluster_bounds = np.array([ cluster_bounds[c] for c in chosen_clusters ])
            
            if not np.isfinite(cluster_bounds).all():
                warn('Infinite lambda values encountered in chosen clusters.'
                     ' This might be due to duplicates in the data.')

            # Extract the plot range of the y-axis and set default center and height values for ellipses.
            # Extremly dense clusters might result in near infinite lambda values. Setting max_height
            # based on the percentile should alleviate the impact on plotting.
            plot_range    = np.hstack([bar_heights, bar_bottoms])
            plot_range    = plot_range[np.isfinite(plot_range)]
            mean_y_center = np.mean([np.max(plot_range), np.min(plot_range)])
            max_height    = np.diff(np.percentile(plot_range, q=[10,90]))

            for c in chosen_clusters:
                c_bounds = cluster_bounds2[c]
                #print("c_bounds: ", c_bounds)
                width  = (c_bounds[1] - c_bounds[0])
                height = (c_bounds[3] - c_bounds[2])
                center = (
                    np.mean([c_bounds[0], c_bounds[1]]),
                    np.mean([c_bounds[3], c_bounds[2]]),
                )
                
                # Set center and height to default values if necessary
                if not np.isfinite(center[1]):
                    center = (center[0], mean_y_center)
                if not np.isfinite(height):
                    height = max_height

                # Ensure the ellipse is visible
                min_height = 0.1*max_height
                if height < min_height:
                    height = min_height

                if selection_palette is not None and \
                        len(selection_palette) >= len(chosen_clusters):
                    oval_color = selection_palette[i]
                else:
                    oval_color = 'r'

                box = Ellipse(
                    center,
                    2.0 * width,
                    1.2 * height,
                    facecolor='none',
                    edgecolor=oval_color,
                    linewidth=2
                )

                if label_clusters:
                    ax.annotate(str(i), xy=center,
                                  xytext=(center[0] - 4.0 * width, center[1] + 0.65 * height),
                                  horizontalalignment='left',
                                  verticalalignment='bottom')

                ax.add_artist(box)

        cb = plt.colorbar(sm, ax=ax)
        cb.ax.set_ylabel('Number of Data Bubbles', fontsize=36)
                                    
        # Cantos do plot
        ax.set_xticks([])
        for side in ('right', 'top', 'bottom'):
            ax.spines[side].set_visible(False)

        ax.invert_yaxis()

        ax.set_ylabel('$\lambda$ value', fontsize=30)

        # Legend
        ax.set_title("Dendrogram", fontsize=34, pad=24)
        #ax.legend(bbox_to_anchor=(0, 1.03, 1, 0.2), loc="lower left", borderaxespad=0, fontsize=28)

        #plt.show()
        
        m_directory = os.path.join(os.getcwd(), "results/dendrograms/dendrograms_t" + str(self.timestamp))
        
        if not os.path.exists(m_directory):
            os.makedirs(m_directory)
                
        fig.savefig("results/dendrograms/dendrograms_t" + str(self.timestamp) + "/minpts_" + str(self.m_minPts) + ".png")
        plt.close()
    
    def getGraphVizString(self):
        newline = "\n"
        tab     = "\t"
        
        sb    = ["graph{" + newline]
        queue = [self.m_root]
        size  = len(queue)
        
        i = 0

        while i < size:
            n = queue[i]
            sb.append(tab + n.getGraphVizString())
            
            numPoint = n.getInternalPoints()
            
            sb.append(str(numPoint) + "}> ")
            
            for v in n.getVertices():
                sb.append(str(v.getID()) + ",")
            
            sb.append("\"];" + newline)
            #sb.append(str(numPoint) + "}\"];" + newline)
            
            children = n.getChildrenMinClusterSize(self.m_minClusterSize)
            #children = n.getChildren()
            
            for child in children:
                numPoint = child.getInternalPoints()
                
                size += 1
                
                queue.append(child)
                    
                sb.append(tab + n.getGraphVizNodeString() + " -- " + child.getGraphVizNodeString())
                sb.append(newline)
            
            i += 1

        sb.append("}")
        
        print(''.join(sb))

class CoreStream(base.Clusterer):

    class BufferItem:
        def __init__(self, x, timestamp, covered):
            self.x = x
            self.timestamp = (timestamp,)
            self.covered = covered
    
    def __init__(
        self,
        m_minPoints            = 10,
        min_cluster_size       = 10,
        decaying_factor: float = 0.25,
        beta:            float = 0.75, 
        mu:              float = 2,
        epsilon:         float = 0.02,
        n_samples_init:  int   = 1000,
        stream_speed:    int   = 100,
        percent                = 0.1,
        method_summarization   = 'single_linkage',
        step                   = 1,
        runtime                = False,
        plot                   = False,
        save_partitions        = False
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
        self.min_cluster_size     = min_cluster_size
        self.step                 = step
        self.method_summarization = method_summarization
        self.runtime              = runtime
        self.plot                 = plot
        self.save_partitions      = save_partitions
        
        # number of clusters generated by applying the variant of DBSCAN algorithm
        # on p-micro-cluster centers and their centers
        self.n_clusters = 0
        
        self.clusters: typing.Dict[int, "DataBubble"]       = {}
        self.p_data_bubbles: typing.Dict[int, "DataBubble"] = {}
        self.o_data_bubbles: typing.Dict[int, "DataBubble"] = {}
        
        self._time_period = math.ceil((1 / self.decaying_factor) * math.log((self.mu * self.beta) / (self.mu * self.beta - 1))) + 1
        print("Time period: ", self._time_period)
        
        if self.method_summarization == 'epsilon':
            self._init_buffer: typing.Deque[typing.Dict] = deque()
        else:
            self._init_buffer = []
        
        self._n_samples_seen = 0
        self.m_update        = None
        
        # DataFrame to save the runtimes
        if self.runtime:
            self.df_runtime_final  = pd.DataFrame(columns=['timestamp', 'data_bubbles', 'summarization', 'mrg', 'mst', 'core_sg', 'multiple_hierarchies'])
            self.df_runtime_stream = pd.DataFrame(columns=['minpts', 'core_sg', 'mst', 'dendrogram', 'selection', 'total'])

        # check that the value of beta is within the range (0,1]
        if not (0 < self.beta <= 1):
            raise ValueError(f"The value of `beta` (currently {self.beta}) must be within the range (0,1].")

    @property
    def centers(self):
        return {k: cluster.getRep(self.timestamp) for k, cluster in self.clusters.items()}

    @staticmethod
    def _distance(point_a, point_b):
        square_sum = 0
        dim        = len(point_a)
        
        for i in range(dim):
            square_sum += math.pow(point_a[i] - point_b[i], 2)
        
        return math.sqrt(square_sum)

    def _get_closest_cluster_key(self, point, clusters):
        min_distance = math.inf
        key          = -1
        
        for k, cluster in clusters.items():
            distance = self.distanceEuclidian(cluster.getRep(self.timestamp), point)
            
            if distance < min_distance and distance <= self.epsilon:
                min_distance = distance
                key          = k
                
        return key

    def distanceEuclidian(self, x1, x2):
        distance = 0
        
        for i in range(len(x1)):
            d        = x1[i] - x2[i]
            distance += d * d
            
        return math.sqrt(distance)
    
    def _merge(self, point):
        # initiate merged status
        merged_status = False

        p = df_bubbles_to_points[(df_bubbles_to_points['x'] == point[0]) & (df_bubbles_to_points['y'] == point[1])]

        if len(self.p_data_bubbles) != 0:
            # try to merge p into its nearest p-micro-cluster c_p
            closest_pdb_key = self._get_closest_cluster_key(point, self.p_data_bubbles)
            
            if closest_pdb_key != -1:
                updated_pdb = copy.deepcopy(self.p_data_bubbles[closest_pdb_key])
                updated_pdb.insert(point, self.timestamp)
                
                if updated_pdb.getExtent(self.timestamp) <= self.epsilon:
                    # keep updated p-micro-cluster
                    self.p_data_bubbles[closest_pdb_key] = updated_pdb

                    df_bubbles_to_points.loc[p.index[0], 'id_bubble'] = closest_pdb_key

                    merged_status = True

        if not merged_status and len(self.o_data_bubbles) != 0:
            closest_odb_key = self._get_closest_cluster_key(point, self.o_data_bubbles)
            
            if closest_odb_key != -1:
                updated_odb = copy.deepcopy(self.o_data_bubbles[closest_odb_key])
                updated_odb.insert(point, self.timestamp)

                if updated_odb.getExtent(self.timestamp) <= self.epsilon:
                    # keep updated o-micro-cluster
                    weight_odb = updated_odb._weight(self.timestamp)
                    
                    if weight_odb > self.mu * self.beta:
                        # it has grown into a p-micro-cluster
                        del self.o_data_bubbles[closest_odb_key]

                        new_key = 0
                        
                        if len(list(self.p_data_bubbles.keys())) == 0:
                            updated_odb.setID(0)
                            self.p_data_bubbles[0] = updated_odb
                        else:
                            new_key = 1
                            
                            while new_key in self.p_data_bubbles:
                                new_key += 1
                            
                            updated_odb.setID(new_key)
                            self.p_data_bubbles[new_key] = updated_odb

                        df_bubbles_to_points.loc[p.index[0], 'id_bubble'] = new_key
                        df_bubbles_to_points['id_bubble']                 = df_bubbles_to_points['id_bubble'].replace((-1) * closest_odb_key, new_key)
                            
                    else:
                        self.o_data_bubbles[closest_odb_key] = updated_odb

                        # Outliers have our key negative
                        df_bubbles_to_points.loc[p.index[0], 'id_bubble'] = (-1) * closest_odb_key
                    
                    merged_status = True
                    
            if not merged_status:
                # create a new o-data_bubble by p and add it to o_data_bubbles
                db_from_p = DataBubble(x=point, timestamp=self.timestamp, decaying_factor=self.decaying_factor)

                key_o = 2

                while key_o in self.o_data_bubbles:
                    key_o += 1
                
                self.o_data_bubbles[key_o] = db_from_p
                
                df_bubbles_to_points.loc[p.index[0], 'id_bubble'] = (-1) * key_o

                merged_status = True

        # when p is not merged and o-micro-cluster set is empty
        if not merged_status and len(self.o_data_bubbles) == 0:
            db_from_p           = DataBubble(x=point, timestamp=self.timestamp, decaying_factor=self.decaying_factor)
            self.o_data_bubbles = {2: db_from_p}

            df_bubbles_to_points.loc[p.index[0], 'id_bubble'] = -2
            
            merged_status = True

    def _is_directly_density_reachable(self, c_p, c_q):
        if c_p._weight(self.timestamp) > self.mu and c_q._weight(self.timestamp) > self.mu:
            # check distance of two clusters and compare with 2*epsilon
            c_p_center = c_p.getRep()
            c_q_center = c_q.getRep()
            distance   = self._distance(c_p_center, c_q_center)
            
            if distance < 2 * self.epsilon and distance <= c_p.calc_radius() + c_q.calc_radius():
                return True
            
        return False

    def _query_neighbor(self, cluster):
        neighbors = deque()
        # scan all clusters within self.p_data_bubbles
        for pmc in self.p_data_bubbles.values():
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

        print("\n>> Timestamp: ", self.timestamp)

        self.time_period_check()
        
        self.data_bubbles_to_points(self.timestamp)
        if self.runtime:
            start_time_total = time.time()
        
        print("> count_potential", len(self.p_data_bubbles))
        print("> count_outlier", len(self.o_data_bubbles))
        
        # (self.m_minPoints / self.mu) is the wort case, when all data bubbles have the slef.mu points
        if len(self.p_data_bubbles) < (self.m_minPoints / self.mu):
            print("no building possible since num_potential_dbs < minPoints")
            return
        
        Vertex.s_idCounter = 0
        
        for db in self.p_data_bubbles.values():
            db.setVertexRepresentative(None)
            db.setStaticCenter(self.timestamp)
        
        G = nx.Graph()
    
        start_mrg = time.time()
        mrg       = MutualReachabilityGraph(G, self.p_data_bubbles.values(), self.m_minPoints, self.timestamp)
        mrg.buildGraph()
        end_mrg   = time.time()
        
        print("> Time for MRG: ", end_mrg - start_mrg)
        
        knng = mrg.getKnngGraph()
        
        start_mst = time.time()
        T         = nx.minimum_spanning_tree(G)
        mst_max   = MinimalSpaningTree(T)
        mst_max.buildGraph()
        end_mst   = time.time()
        
        print("> Time for MST:", end_mst - start_mst)
        
        csg = CoreSG(mst_max, knng, self.timestamp)
        
        if self.runtime:
            self.df_runtime_final.at[self.timestamp, 'core_sg'] = (time.time() - start_time_total)
        
        print("Computing Multiple Hierarchies:")
        start_hierarchies = time.time()
        self.computeMulipleHierarchies(csg, 2, self.m_minPoints)
        end_hierarchies   = time.time()
        
        if self.runtime:
            self.df_runtime_final.at[self.timestamp, 'data_bubbles']         = len(self.p_data_bubbles)
            self.df_runtime_final.at[self.timestamp, 'timestamp']            = self.timestamp
            self.df_runtime_final.at[self.timestamp, 'mrg']                  = end_mrg - start_mrg
            self.df_runtime_final.at[self.timestamp, 'mst']                  = end_mst - start_mst
            self.df_runtime_final.at[self.timestamp, 'multiple_hierarchies'] = end_hierarchies - start_hierarchies
        
        print("> Time for Multiple Hierarchies: ", end_hierarchies - start_hierarchies)
        
        print("\n****************************************************************************************")
        
        self.m_update = Updating(mrg, mst_max, csg)
        self.mst      = mst_max
        
        # Plot MRG      -> mrg.buildAbsGraph()
        # Plot KNNG     -> knng.buildAbsGraph(self.timestamp)
        # Plot MST Max  -> mst_max.buildAbsGraph(self.timestamp)
        # GraphViz MRG  -> mrg.getGraphVizString()
        # GraphViz MST  -> mst_max.getGraphVizString()
        # GraphViz KNNG -> knng.getGraphVizString()
        # GraphViz CSG  -> csg.getGraphVizString()

    def computeMulipleHierarchies(self, csg, min_pts_min, min_pts_max):

        df_partition = df_bubbles_to_points[(df_bubbles_to_points['id_bubble'] != -1)]
        len_points   = df_partition.shape[0]
        len_dbs      = len(csg.getVertices())
        
        if self.save_partitions:
            matrix_partitions         = [[-1 for j in range(len_dbs + 10)] for i in range(self.m_minPoints + 10)]
            matrix_partitions_bubbles = [[-1 for j in range(len_points + 10)] for i in range(self.m_minPoints + 10)]
            matrix_partitions_hdbscan = [[-1 for j in range(len_points + 10)] for i in range(self.m_minPoints + 10)]

        for minpts in range(min_pts_max, min_pts_min - 1, -self.step):
            
            start_time_total = time.time()
            
            print("\n-------------------------------------------------------------------------------------")
            print("MinPts: " + str(minpts))
            
            start_hierarchy = time.time()
            csg.computeHierarchyMinPts(minpts)
            end_hierarchy   = time.time()
            
            print("> Time for CORE-SG edges: ", end_hierarchy - start_hierarchy)
            
            # MST
            start_mst   = time.time()
            G           = csg.getGraphNetworkx()
            T           = nx.minimum_spanning_tree(G, weight='weight')
            mst_csg     = MinimalSpaningTree(T)
            mst_csg.buildGraph()
            end_mst     = time.time()
            
            print("> Time for MST: ", end_mst - start_mst)
            
            self.m_update = Updating(None, mst_csg, csg)
            self.mst_mult = mst_csg
            
            # Dendrogram
            start_dendrogram = time.time()
            dendrogram       = Dendrogram(self.m_update.getMST(), self.min_cluster_size, minpts, self.timestamp) # MST, miClusterSize, minPts
            dendrogram.build()
            end_dendrogram   = time.time()
            
            print("> Time for dendrogram: ", end_dendrogram - start_dendrogram)
            
            # Time Selection Clusters
            start_selection = time.time()
            selection       = dendrogram.clusterSelection()

            # Partitions Corestream MinPts
            cont             = 1
            partition_bubble = {}

            if self.save_partitions:
                for n in selection:
                    it = iter(n.getVertices())

                    for el in it:
                        partition_bubble[el.getDataBubble().getID()] = cont
                        matrix_partitions[minpts][el.getID()]        = cont

                    cont += 1

                cont = 0
                for i, row in df_partition.iterrows():
                    if row['id_bubble'] in partition_bubble:
                        matrix_partitions_bubbles[minpts][cont] = partition_bubble[row['id_bubble']]
                    cont += 1
            
            end_selection = time.time()
            print("> Time for Selection Clusters: ", end_selection - start_selection)
            
            # Partitions HDBSCAN
            clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=minpts, match_reference_implementation = True)
            clusterer.fit(df_partition[['x', 'y']])
            labels = clusterer.labels_
            
            p = 0
            for i in labels:
                matrix_partitions_hdbscan[minpts][p] = i
                p += 1

            # Plot HDBSCAN result
            if self.plot:
                self.plot_hdbscan_result(minpts, labels, df_partition)
            
            # Runtime
            if self.runtime:
                self.df_runtime_stream.at[i, 'minpts']     = minpts
                self.df_runtime_stream.at[i, 'core_sg']    = end_hierarchy - start_hierarchy
                self.df_runtime_stream.at[i, 'mst']        = end_mst - start_mst
                self.df_runtime_stream.at[i, 'dendrogram'] = end_dendrogram - start_dendrogram
                self.df_runtime_stream.at[i, 'selection']  = end_selection - start_selection
                self.df_runtime_stream.at[i, 'total']      = time.time() - start_time_total
            
            # GraphViz CORE-SG    -> csg.getGraphVizString(self.timestamp, i)
            # GraphViz MST        -> mst_csg.getGraphVizString(self.timestamp, i)
            # Plot MST            -> mst_csg.buildAbsGraph(self.timestamp)
            # GraphViz Dendrogram -> if i == 200: dendrogram.getGraphVizString()
        
        if self.save_partitions:
            self.save_partitions_bubble_and_points_minpts(len_points, matrix_partitions_bubbles, matrix_partitions_hdbscan, min_pts_min, min_pts_max) 
            self.save_partitions_final(matrix_partitions, len_dbs, csg.getVertices(), min_pts_min, min_pts_max)
        if self.plot:
            self.plot_partitions(matrix_partitions, len_dbs, min_pts_min, min_pts_max, df_partition)
        
        # reset df_bubbles_to_points['id_bubble'] in timestamp
        df_bubbles_to_points['id_bubble'] = -1
    
    def _expand_cluster(self, db, neighborhood):
        for idx in neighborhood:
            item = self._init_buffer[idx]
            
            if not item.covered:
                item.covered = True
                db.insert(item.x, self.timestamp)

    def _get_neighborhood_ids(self, item):
        neighborhood_ids = deque()
        
        for idx, other in enumerate(self._init_buffer):
            if not other.covered:
                #print(">> ", self._distance(item.x, other.x))
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
                    db = DataBubble(
                        x=item.x,
                        timestamp=self.timestamp,
                        decaying_factor=self.decaying_factor,
                    )
                    self._expand_cluster(db, neighborhood)
                    db.setStaticCenter(self.timestamp)
                    self.p_data_bubbles.update({len(self.p_data_bubbles): db})
                else:
                    item.covered = False
                    
        end = time.time()
        
        self.df_runtime_final.at[self.timestamp, 'summarization'] = end - start if self.runtime else None
        
        print("> Time for Summarization: ", end - start)
    
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
        min_db = max_db = 0
        epsilon         = {}
        
        for i in range(len_buffer):
            
            labels_visited[labels[i]] += 1
            
            if labels_visited[labels[i]] == 1:
                db = DataBubble(
                    x = dict(zip([j for j in range(len(self._init_buffer[i]))], self._init_buffer[i])),
                    timestamp=self.timestamp,
                    decaying_factor=self.decaying_factor,
                )

                db.setID(labels[i])
                
                self.p_data_bubbles.update({labels[i]: db})
                
            else:
                self.p_data_bubbles[labels[i]].insert(dict(zip([j for j in range(len(self._init_buffer[i]))], self._init_buffer[i])), self.timestamp)

                if labels_visited[labels[i]] == self.mu:
                    count_potential += 1
                if self.p_data_bubbles[labels[i]].getN() >= self.mu:
                    epsilon[labels[i]] = self.p_data_bubbles[labels[i]].getExtent(self.timestamp)
                
            max_db = max(max_db, self.p_data_bubbles[labels[i]].getN())

            point = df_bubbles_to_points[(df_bubbles_to_points['x'] == self._init_buffer[i][0]) & (df_bubbles_to_points['y'] == self._init_buffer[i][1])]
            df_bubbles_to_points.at[point.index[0], 'id_bubble'] = labels[i]
        
        # outliers data_bubbles
        if count_potential != len(self.p_data_bubbles):
            key   = 0
            key_p = 0
            key_o = 2
            
            while labels_visited[key]:
                if labels_visited[key] < self.mu:
                    df_bubbles_to_points['id_bubble'] = df_bubbles_to_points['id_bubble'].replace(key, (-1) * key_o)
                    
                    self.o_data_bubbles[key_o] = self.p_data_bubbles[key]
                    self.p_data_bubbles.pop(key)

                    key_o += 1
                else:
                    if key != key_p:
                        self.p_data_bubbles[key].setID(key_p)
                        self.p_data_bubbles[key_p] = self.p_data_bubbles.pop(key)

                        if min_db == 0:
                            min_db = self.p_data_bubbles[key_p].getN()
                        else:
                            min_db = min(min_db, self.p_data_bubbles[key_p].getN())
                        
                        #update new key
                        df_bubbles_to_points['id_bubble'] = df_bubbles_to_points['id_bubble'].replace(key, key_p)

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
        
        print("> Total: ", (count_potential + len(self.o_data_bubbles)))
        print("> Bubbles Potential: ", count_potential)
        print("> Min_DB: ", min_db)
        print("> Max_DB: ", max_db)
        print("> Time for DBs: ", end - start)
        print("> Epsilon min: ", e_min)
        print("> Epsilon mean: ", e_mean)
        print("> Epsilon max: ", e_max)

    def time_period_check(self):
        # Periodic cluster removal
            
        for i, p_data_bubble_i in list(self.p_data_bubbles.items()):
            if p_data_bubble_i._weight(self.timestamp) < self.mu * self.beta:
                # c_p became an outlier and should be deleted
                self.p_data_bubbles.pop(i)

        for j, o_data_bubble_j in list(self.o_data_bubbles.items()):
            # calculate xi
            xi = (2**(-self.decaying_factor * (self.timestamp - o_data_bubble_j.creation_time + self._time_period)) - 1) / (2 ** (-self.decaying_factor * self._time_period) - 1)

            if o_data_bubble_j._weight(self.timestamp) < xi:
                # c_o might not grow into a p-micro-cluster, we can safely delete it
                self.o_data_bubbles.pop(j)
        
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
                self.save_runtime_timestamp() if self.runtime else None
                
                self.initialized = True
                
                del self._init_buffer
                
            return self

        # Merge
        self._merge(x)

        if self.timestamp > 0 and self.timestamp % self._time_period == 0:
            self.time_period_check()
        
        return self

    def predict_one(self, sample_weight=None):        
        # This function handles the case when a clustering request arrives.
        # implementation of the DBSCAN algorithm proposed by Ester et al.
        
        if not self.initialized:
            # The model is not ready
            return 0
        
        self._build()
            
    def save_partitions_final(self, matrix_partitions, count_DB, vertices, min_pts_min, min_pts_max):
        m_directory = os.path.join(os.getcwd(), "results/flat_solutions")
        
        try:
            sub_dir = os.path.join(m_directory, "flat_solution_partitions_t" + str(self.timestamp))

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            for i in range(min_pts_max, min_pts_min - 1, -self.step):
                with open(os.path.join(sub_dir, "CoreStream_Partitions_MinPts_" + str(i) + ".csv"), 'w') as writer:
                    writer.write("x,y,N,radio,color,cluster,ID\n")

                    cores = ["blue", "red", "orange", "green", "purple", "brown", "pink", "olive", "cyan"]

                    for v in vertices:
                        if matrix_partitions[i][v.getID()] == -1:
                            writer.write(str(v.getDataBubble().getRep(self.timestamp)[0]) + "," + str(v.getDataBubble().getRep(self.timestamp)[1]) + "," + str(v.getDataBubble()._weight(self.timestamp)) + "," + str(v.getDataBubble().getExtent(self.timestamp)) + ",black,-1" + "," + str(v.getDataBubble().getID()) + "\n")
                        else:
                            writer.write(str(v.getDataBubble().getRep(self.timestamp)[0]) + "," + str(v.getDataBubble().getRep(self.timestamp)[1]) + "," + str(v.getDataBubble()._weight(self.timestamp)) + "," + str(v.getDataBubble().getExtent(self.timestamp)) + "," + cores[matrix_partitions[i][v.getID()] % 9] + "," + str(matrix_partitions[i][v.getID()]) + "," + str(v.getDataBubble().getID()) + "\n")

            # Saving the partitions bubbles
            with open(os.path.join(sub_dir, "CoreStream_All_Partitions_MinPts.csv"), 'w') as writer:
                for j in range(count_DB):
                    if j == 0:
                        writer.write(str(j))
                    else:
                        writer.write(", " + str(j))

                writer.write("\n")

                for i in range(min_pts_max, min_pts_min - 1, -self.step):
                    for j in range(count_DB):
                        if j == 0:
                            writer.write(str(matrix_partitions[i][j]))
                        else:
                            writer.write(", " + str(matrix_partitions[i][j]))

                    writer.write("\n")

        except FileNotFoundError as e:
            print(e)
            
    def plot_partitions(self, matrix_partitions, count_DB, min_pts_min, min_pts_max, df_partition):
        sns.set_context('poster')
        sns.set_style('white')
        sns.set_color_codes()
        
        plot_kwds = {'s' : 3, 'linewidths':0}
        
        m_directory = os.path.join(os.getcwd(), "results/plots")
        
        try:
            sub_dir = os.path.join(m_directory, "plot_bubbles_t" + str(self.timestamp))

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
        
            for i in range(min_pts_max, min_pts_min - 1, -self.step):
                partition = pd.read_csv('results/flat_solutions/flat_solution_partitions_t' + str(self.timestamp) + '/CoreStream_Partitions_MinPts_' + str(i) + '.csv', sep=',')

                # Statistic partition-------------------------------
                count_outlier = 0
                count_cluster = 0

                for j in range(len(partition)):

                    if(partition['cluster'].loc[j] == -1):
                        count_outlier += 1

                    if(partition['cluster'].loc[j] > count_cluster):
                        count_cluster = partition['cluster'].loc[j]

                legend  = ""
                legend += "MinPts: " + str(i) + "  "
                legend += "| Outliers: " + str(int((count_outlier * 100.0) / len(partition))) + "%  "
                legend += "| Clusters: " + str(count_cluster) + "  "
                legend += "| DBs: " + str(len(partition)) + "  "
                legend += "| Timestamp: " + str(self.timestamp)
                # -------------------------------------------------

                plt.figure(figsize = (16,12))

                for j in range(len(partition)):
                    #plt.plot([partition['x'].loc[j]], [partition['y'].loc[j]], 'o', ms=partition['radio'].loc[j] * 50, mec=partition['color'].loc[j], mfc='none', mew=2)
                    plt.gca().add_patch(plt.Circle((partition['x'].loc[j], partition['y'].loc[j]), partition['radio'].loc[j], color=partition['color'].loc[j], fill=False))
                    #plt.text(partition['x'].loc[j], partition['y'].loc[j], str(partition['ID'].loc[j]), fontsize=10, ha='center', va='center')
            
                #start = ((self._n_samples_seen / 7000) - 1) * self.n_samples_init
                #end   = start + self.n_samples_init
                #plt.scatter(data[int(start):int(end), 0], data[int(start):int(end), 1], **plot_kwds, label=legend)
                
                plt.scatter(df_partition['x'], df_partition['y'], **plot_kwds, label=legend)
                plt.legend(bbox_to_anchor=(-0.1, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, fontsize=26)
                plt.savefig("results/plots/plot_bubbles_t" + str(self.timestamp) + "/minpts_" + str(i) + ".png")
                plt.close('all')
                
        except FileNotFoundError as e:
            print(e)

    def plot_hdbscan_result(self, minpts, labels, df_partition):
        m_directory = os.path.join(os.getcwd(), "results/plots")
        
        sub_dir = os.path.join(m_directory, "plot_bubbles_t" + str(self.timestamp))

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        
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

        plt.scatter(df_partition['x'], df_partition['y'], c=labels, cmap='magma', **plot_kwds)

        plt.savefig("results/plots/plot_bubbles_t" + str(self.timestamp) + "/minpts _" + str(minpts) + "_hdbscan.png")
        
        plt.close('all')
    
    def save_partitions_bubble_and_points_minpts(self, len_partitions, matrix_partitions_bubble, matrix_partitions_hdbscan, min_pts_min, min_pts_max):

        m_directory = os.path.join(os.getcwd(), "results/flat_solutions")
        
        try:
            sub_dir = os.path.join(m_directory, "flat_solution_partitions_t" + str(self.timestamp))

            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

            # Partitions Corestream MinPts
            with open(os.path.join(sub_dir, "all_partitions_bubbles.csv"), 'w') as writer:
                for j in range(len_partitions):
                    if j == 0:
                        writer.write(str(j))
                    else:
                        writer.write(", " + str(j))

                writer.write("\n")

                for i in range(min_pts_min, min_pts_max + 1, self.step):
                    for j in range(len_partitions):
                        if j == 0:
                            writer.write(str(matrix_partitions_bubble[i][j]))
                        else:
                            writer.write(", " + str(matrix_partitions_bubble[i][j]))

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
                writer.write("minpts,core_sg,mst,dendrogram,selection,total\n")

                for _, linha in self.df_runtime_stream.iterrows():
                    writer.write(str(linha['minpts']) + ',' + str(linha['core_sg']) + ',' + str(linha['mst']) + ',' + str(linha['dendrogram']) + ',' + str(linha['selection']) + ',' + str(linha['total']) + "\n")

        except FileNotFoundError as e:
            print(e)
            
    def save_runtime_final(self):
        m_directory = os.path.join(os.getcwd(), "results/runtime")
        
        try:
            if not os.path.exists(m_directory):
                os.makedirs(m_directory)

            with open(os.path.join(m_directory, "runtime_final_t" + str(self.timestamp) + ".csv"), 'w') as writer:
                writer.write("timestamp,data_bubbles,summarization,mrg,mst,core_sg,multiple_hierarchies\n")

                for _, linha in self.df_runtime_final.iterrows():
                    writer.write(str(linha['timestamp']) + ',' + str(linha['data_bubbles']) + ',' + str(linha['summarization']) + ',' + str(linha['mrg']) + ',' + str(linha['mst']) + ',' + str(linha['core_sg']) + ',' + str(linha['multiple_hierarchies']) + "\n")

        except FileNotFoundError as e:
            print(e)

    def data_bubbles_to_points(self, timestamp):
        
        m_directory = os.path.join(os.getcwd(), "results/datasets")
        
        try:
            if not os.path.exists(m_directory):
                os.makedirs(m_directory)

                sns.set_context('poster')
                sns.set_style('white')
                sns.set_color_codes()

                plot_kwds = {'s' : 1, 'linewidths':0}

                plt.figure(figsize=(12, 10))

                for i, row in df_bubbles_to_points.iterrows():
                    if row['id_bubble'] not in self.p_data_bubbles and row['id_bubble'] > -1:
                        df_bubbles_to_points.at[i, 'id_bubble'] = -1
                    elif ((-1) * row['id_bubble']) not in self.o_data_bubbles and row['id_bubble'] < -1:
                        df_bubbles_to_points.at[i, 'id_bubble'] = -1

                for key, value in self.p_data_bubbles.items():
                    plt.gca().add_patch(plt.Circle((value.getRep(timestamp)[0], value.getRep(timestamp)[1]), value.getExtent(timestamp), color='red', fill=False))

                for key, value in self.o_data_bubbles.items():
                    plt.gca().add_patch(plt.Circle((value.getRep(timestamp)[0], value.getRep(timestamp)[1]), value.getExtent(timestamp), color='blue', fill=False))

                df_bubbles_to_points[(df_bubbles_to_points['id_bubble'] != -1)].to_csv('results/datasets/data_t' + str(self.timestamp) + '.csv', index=False)

                df_plot = df_bubbles_to_points[(df_bubbles_to_points['id_bubble'] != -1)]

                cmap = plt.get_cmap('tab10', len(list(set([row['id_bubble'] for i, row in df_plot.iterrows()]))))

                plt.title("Timestamp: " + str(self.timestamp) + " | # Points: " + str(df_plot.shape[0]) + " | # DBs: " + str(len(self.p_data_bubbles)), fontsize=20)
                plt.scatter(df_plot['x'], df_plot['y'], c='green', **plot_kwds)
                plt.savefig("results/datasets/plot_dataset_t" + str(self.timestamp) + ".png")
                plt.close()
        except FileNotFoundError as e:
            print(e)
if __name__ == "__main__":

    data = pd.read_csv(sys.argv[1], sep=',')

    #variável de pontos iniciais
    initial_points = int(sys.argv[2])

    scaler = MinMaxScaler()

    scaler.fit(data)

    data = pd.DataFrame(data=scaler.transform(data))

    df_bubbles_to_points = data.copy()
    df_bubbles_to_points['id_bubble'] = -1

    data = data.to_numpy()

    #denstream = CoreStream(m_minPoints= int(sys.argv[2]) , n_samples_init= int(sys.argv[3]), epsilon= float(sys.argv[4]))   

    corestream = CoreStream(int(sys.argv[3]),
                        min_cluster_size = 25,
                        step=2,
                        decaying_factor=0.025,
                        mu=2, n_samples_init=initial_points, 
                        epsilon=0.005,
                        percent=0.15,
                        method_summarization='single',
                        stream_speed=100,
                        runtime=False,
                        plot=False,
                        save_partitions=True)

    count_points = 0

    for x, _ in stream.iter_array(data):
        denstream = corestream.learn_one(x)
        
        count_points += 1
        
        if not (count_points % initial_points) and count_points != initial_points:
            corestream.predict_one()
            #corestream.save_runtime_timestamp()
    #corestream.save_runtime_final()
