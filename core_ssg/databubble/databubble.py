import math

from abc import ABCMeta

class DataBubble(metaclass=ABCMeta):
  
    def __init__(self, x, timestamp, decaying_factor):

        self.x = x
        self.last_edit_time = timestamp
        self.creation_time = timestamp
        self.decaying_factor = decaying_factor

        self.N = 1
        self.linear_sum = x
        self.squared_sum = {i: (x_val * x_val) for i, x_val in x.items()}
        self.m_staticCenter = len(self.linear_sum)

    def calc_norm_cf1_cf2(self):
        # |CF1| and |CF2| in the paper
        x1 = 0
        x2 = 0
        res = 0
        weight = self._weight()
        for key in self.linear_sum.keys():
            val_ls = self.linear_sum[key]
            val_ss = self.squared_sum[key]
            x1 = 2 * val_ss * weight
            x2 = 2 * val_ls**2            
            diff = (x1 - x2) / (weight * (weight - 1))
            res += math.sqrt(diff) if diff > 0 else 0
        # return |CF1| and |CF2|
        return res
    def getN(self):
        return self.N

    def calc_weight(self):
        return self._weight()

    def _weight(self):
        return self.N
        
    def getRep(self):  
        weight = self._weight()
        center = {key: (val) / weight for key, val in self.linear_sum.items()}
        return center

    def getExtent(self):        
        res = self.calc_norm_cf1_cf2()
        return res

    def insert(self, x):
        self.N += 1
        
        for key, val in x.items():
            try:
                self.linear_sum[key] += val
                self.squared_sum[key] += val * val
            except KeyError:
                self.linear_sum[key] = val
                self.squared_sum[key] = val * val

    def merge(self, cluster):
        self.N += cluster.N
        for key in cluster.linear_sum.keys():
            try:
                self.linear_sum[key] += cluster.linear_sum[key]
                self.squared_sum[key] += cluster.squared_sum[key]
            except KeyError:
                self.linear_sum[key] = cluster.linear_sum[key]
                self.squared_sum[key] = cluster.squared_sum[key]
        if self.last_edit_time < cluster.creation_time:
            self.last_edit_time = cluster.creation_time

    def getNnDist(self, k):
        return ((k/self.N)**(1.0/len(self.linear_sum)))*self.getExtent()

    def fading_function(self, time):
        return 2 ** (-self.decaying_factor * time)
    

    def getVertexRepresentative(self):
        return self.m_vertexRepresentative
    
    def getStaticCenter(self):
        return self.m_staticCenter

    def setStaticCenter(self):
        m_static_center = self.getRep().copy()
        return m_static_center