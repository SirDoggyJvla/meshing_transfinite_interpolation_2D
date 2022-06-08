# Abstract
"""
S. Cadet
19/04/2022
-- Tools MeshCPU Mesh Generation
    -- Main
"""
# Setting working directory
if __name__=='__main__':
    import os
    path = __file__
    path = path.replace("\\","/")
    path = path.replace("/toolsRef.py","")
    path_main = path.replace("/src","")
    os.chdir(path)

# Imports
import numpy as np
import matplotlib.pyplot as plt
from math import *

import sys, os, time, inspect
from pyDOE.doe_lhs import lhs
import time, datetime
import code   #code.interact(local=dict(globals(), **locals()))
from itertools import cycle
import scipy.stats as ss
from six.moves import input
from sklearn.cluster import KMeans
from copy import deepcopy

# HiDPI Matplotlib
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

import generalFunctions as gF


# =============================================================================
# Class Properties for meshing
# =============================================================================
class Properties():
    """
    This class is made to be used in the other class MeshCPU and Mesh
    """
    # Properties
    @property
    def N(self): 
        """ Total number of Nodes """
        return self.n * self.m
    
    @property
    def N_el(self): 
        """ Total number of Elements """
        return (self.n-1)*(self.m-1)
    
    @property
    def N_ed_hor(self):
        """ Total number of horizontal Edges """
        return (self.n-1)*self.m
    
    @property
    def N_ed_ver(self):
        """ Total number of vertical Edges """
        return (self.m-1)*self.n
   
    @property
    def N_ed(self):
        """ Total number of Edges """
        return self.N_ed_hor + self.N_ed_ver


    # Export n or m if needed
    def export_n(self): return self.n
    def export_m(self): return self.m
    
    
    # Export number of nodes, elements or edges
    def export_N(self): return self.N
    def export_N_el(self): return self.N_el
    def export_N_ed(self): return self.N_ed
    


# =============================================================================
# Class MeshCPU for reference mesh
# =============================================================================
class MeshCPU(gF.ClassBasic,Properties):
    """
    --- Returns reference (CPU) mesh, a 1x1 square divided vertically in n segments and horizontally in m segments
    
        For n=4, m=3
     ↑  8---9---10--11            o-6-o-7-o-8-o                ↑  ┌─┬─┬─┐
     |  | 3 | 4 | 5 <--- Element  13  14  15  16    Result     |  │ │ │ │
   m |  4---5---6---7             o-3-o-4-o-5-o    =======>  1 |  ├─┼─┼─┤
     |  | 0 | 1 | 2 |             9   10  11  12 _             |  │ │ │ │
     ↓  0---1---2---3 _           o-0-o-1-o-2-o |\             ↓  └─┴─┴─┘
                     |\                           \
        <----------->  \                           Edge           <----->
               n        Node                                         1

               
    Nodes k from 0 to (n*m) -1 = 11 (in this case)
    Elements K from 0 to (n-1)*(m-1) -1 = 5 (in this case)
    Edges from 0 to (n-1)*m + n*(m-1) = 16 (in this case)

    A = MeshCPU(input1,input2...)
    """
    # Initialization
    def __init__(self,n=None,m=None,build=True):
        """
        --- Initialization
        A = MeshCPU(n=,m=)
        or
        A = MeshCPU(n=,m=,build=False)
        
        -- Input
        n               : int       n-values - i length
        m               : int       m-values - j length
        build           : boolean   Automatically create mesh values ?         (default: True)
        """
        # Data
        self.n = n
        self.m = m
        
        # Objects
        self.X_ref = None
        self.elem = None
        self.edges = None
        self.listTypeEdges = None
        self.X = None
        self.edgesPos = None
        self.edgesHor = None
        self.edgesVer = None
        
        # # List
        # self.listNodesNodes = None
        # self.listNodesElements = None
        # self.listNodesEdges = None
        # self.listEdgesElements = None
        # self.listEdgesEdges = None
        # self.listElementsEdges = None
        # self.listElementsElements = None
        if build: self.build()
        return
    
    
    # Export X_ref, elem or edges
    """
    -- Input
    build               : boolean   Rebuilds the array to export before exporting it (default : false)
    -- Output           :
        - X_ref         : (N,2) array       coordinates of CPU nodes
        - elem          : (N_el,4) array    nodes of elements
        - edges         : (N_ed,2) array    nodes of edges
    """
    def export_X_ref(self,build=False): 
        if build: self.generateNodes()
        return self.X_ref
    def export_elem(self,build=False): 
        if build: self.generateElements()
        return self.elem
    def export_edges(self,build=False):
        if build: self.generateEdges()
        return self.edges
    
    
    # Updates the value of n or m
    def update(self,n=None,m=None,build=True):
        """
        --- Updates n and m values and rebuild the CPU mesh if "build=True"
        """
        if n is not None: self.n = n
        if m is not None: self.m = m
        if build: self.build()
        return
    
    # Set X as X_ref to use in import export
    def setXasCPU(self):
        self.X = self.X_ref
        return 
    
    # Main function
    def build(self):
        self.generateNodes()
        self.generateElements()
        self.generateEdges()
        return
       
    
    # Generates coordinates of Nodes
    def generateNodes(self):
        """
        --- Computes position of Nodes of reference mesh
        X_ref = A.generateNodes()
        
        -- Output
        X_ref           : (n,2) array
        x=X[:,0], y=X[:,1]
        """
        self.X_ref = np.zeros((self.N,2))
        for i in range(0,self.n):
            for j in range(0,self.m):
                k = (i) + (self.n) * (j)
                self.X_ref[k,0]=i/(self.n-1)
                self.X_ref[k,1]=j/(self.m-1)
        return self.X_ref
    
    
    # Generates elements
    def generateElements(self):
        """
        --- Determines the elements from 0 to N_el-1 and their nodes k (4 for each elements)
        A.generateElements()
    
        -- Output
        elem            : (N_el,4)  array of Nodes of each elements
        """
        self.generateNodes()
        self.elem = np.zeros((self.N_el,4))
        
        inc = 0
        for j in range(0,self.N_el):
            self.elem[j,0] = j + inc
            self.elem[j,1] = j + 1 + inc
            self.elem[j,2] = j + self.n + 1 + inc
            self.elem[j,3] = j + self.n + inc
            if j == ((self.n * (inc + 1) - inc ) - 2):
                inc += 1
        
        self.elem = self.elem.astype(int)
        return self.elem
    
    
    # Generates Edges
    def generateEdges(self):
        """
        --- Determines the edges from 0 to N_ed-1 and their nodes k (2 for each edges)
        A.generateEdges(self)
        
        -- Ouput
        edges           : (N_ed,2)  array of Nodes of each edges
            - the first edge is an horizontal edge, the last one is vertical
            - the first part is horizontal edges, last is vertical edges
            - the position of the switch between horizontal and vertical is obtained with A.determineEdgesPos()
            - first value of array is left (for horizontal edge), or bottom (for vertical edge) node from edge
            - second value is right or top node
            
        -- Complementary operation
        listTypeEdges   : list of size N_ed
            - from edge 0 to N_ed, gives its positions on the mesh
            - left, bottom, right, top for outside edges of the mesh
            - interior for inside edges of the mesh
            - can be obtained via A.export_listTypeEdges()
        """
        self.edges = np.zeros((self.N_ed,2))
        i = 0
        
        # Horizontal Edges
        for p in range(0,self.m):
            for k in range( self.n*p, self.n*(p+1) - 1 ):
                self.edges[i,0] = k
                self.edges[i,1] = k+1
                i += 1
        
        # Vertical Edges
        for p in range(0,self.m-1):
            for k in range( self.n*p, self.n*(p+1) ):
                self.edges[i,0] = k
                self.edges[i,1] = k+self.n
                i += 1
        
        self.edges = self.edges.astype(int)
        self.determineEdgesPos()
        
        # Edges type
        self.listTypeEdges = []
        for i in range(0,self.N_ed):
            a = 0
            if i < self.edgesPos:
                for j in range(0,self.n-1):
                    if j == self.edges[i,0] and j+1 == self.edges[i,1]:
                        self.listTypeEdges.append('bottom')
                        a = 1
                    node = self.n*(self.m-1)
                    if node + j == self.edges[i,0] and node + j + 1 == self.edges[i,1]:
                        self.listTypeEdges.append('top')
                        a = 2
            elif i >= self.edgesPos:
                for j in range(0,self.m):
                    if self.n*j == self.edges[i,0] and self.n*j + self.n == self.edges[i,1]:
                        self.listTypeEdges.append('left')
                        a = 3
                    if self.n*(j+1) - 1 == self.edges[i,0] and self.n*(j+1) + self.n - 1 == self.edges[i,1]:
                        self.listTypeEdges.append('right')
                        a = 4
            if a == 0: self.listTypeEdges.append('interior')
        return self.edges
    
    def determineEdgesPos(self):
        """ Gives the position of the start of the vertical edges from the edge array """
        for i in range(0,self.N_ed):
            if self.edges[i,0] == 0:
                self.edgesPos = i
        self.edgesHor = self.edges[:self.edgesPos]
        self.edgesVer = self.edges[self.edgesPos:]
        return self.edgesPos 
        
    
    def nodePosition(self):
        """
        --- For every Node k, gives i and j, it's horizontal and vertical position
        A.nodePosition()
        
        -- Output
        nodePos         : (N,2)
        """
        nodePos = np.zeros((self.N,2))
        for k in range(0,self.N):
            nodePos[k,1] = int(k/(self.n))
            nodePos[k,0] = k - nodePos[k,1] * (self.n)
        nodePos = nodePos.astype(int)
        return nodePos
    
    def nodePositionInv(self,i,j):
        """
        --- Gives a Node number k by knowing i and j, its horizontal and vertical position
        A.nodePositionInv(i=,j=)
        
        -- Input
        i, j            : int       horizontal and vertical position
        
        -- Output
        k               : int       value of k for i and j
        """
        k = i + self.n * j
        return int(k)
    
    
    # Lists Nodes
    def generateListNodesNodes(self):
        """
        --- Creates the lists of every Nodes linked by an edge with the Nodes, for every Nodes
        A.generateListNodesNodes(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesEdges         : list
            - first term is first Node, last term is last Node
            - rotation rule start: left
        """
        self.listNodesNodes = []
        for i in range(0,self.N):
            temp = []
            pos1 = 0
            pos2 = 0
            for j in range(0,len(self.edgesHor)):
                if self.edgesHor[j,1] == i:
                    temp.append(self.edgesHor[j,0])
                    pos1 = j + 1
            for j in range(0,len(self.edgesVer)):
                if self.edgesVer[j,1] == i:
                    temp.append(self.edgesVer[j,0])
                    pos2 = j + 1
            for j in range(pos1,len(self.edgesHor)):
                if self.edgesHor[j,0] == i:
                    temp.append(self.edgesHor[j,1])
            for j in range(pos2,len(self.edgesVer)):
                if self.edgesVer[j,0] == i:
                    temp.append(self.edgesVer[j,1])
            self.listNodesNodes.append(temp)
        return self.listNodesNodes
    
    def generateListNodesEdges(self):
        """
        --- Creates the lists of every Edges linked to the Nodes, for every Nodes
        A.generateListNodesEdges(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesEdges         : list
            - first term is first Node, last term is last Node
            - rotation rule start: left
        """
        self.listNodesEdges = []
        for i in range(0,len(self.X_ref)):
            temp = []
            for j in range(0,len(self.edges)-self.edgesPos):
                if self.edges[j,1] == i:
                    temp.append(j)
            for j in range(len(self.edges)-self.edgesPos,len(self.edges)):
                if self.edges[j,1] == i:
                    temp.append(j)
            for j in range(0,len(self.edges)-self.edgesPos):
                if self.edges[j,0] == i:
                    temp.append(j)
            for j in range(len(self.edges)-self.edgesPos,len(self.edges)):
                if self.edges[j,0] == i:
                    temp.append(j)
            self.listNodesEdges.append(temp)
        return self.listNodesEdges
        
    def generateListNodesElements(self):
        """
        --- Creates the lists of every elements surrounding the Nodes, for every Nodes
        A.generateListNodesElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesElements        : list
            - first term is first Node, last term is last Node
            - rotation rule start: bottom left
        """
        self.listNodesElem = []
        for i in range(0,len(self.X_ref)):
            temp = []
            for j in range(0,len(self.elem)):
                if self.elem[j,2] == i:
                    temp.append(j)
                if self.elem[j,3] == i:
                    temp.append(j)
                if self.elem[j,0] == i:
                    temp.append(j)
                if self.elem[j,1] == i:
                    temp.append(j)
            if len(temp) == 4:
                a = temp[2]
                b = temp[3]
                temp[2] = b
                temp[3] = a
            self.listNodesElem.append(temp)
        return self.listNodesElem


    # Lists Edges
    def generateListEdgesElements(self):
        """
        --- Creates the lists of every elements surrounding the edges, for every edges
        A.generateListEdgesElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listEdgesElements      : list
            - first term is first edge, last term is last edge, with the horizontal and vertical edges keeping the same order as self.edges
            - rotation rule start: bottom left
        """
        self.listEdgesElements = []
        for i in range(0,len(self.edgesHor)):
            temp = []
            for j in range(0,len(self.elem)):
                if self.edgesHor[i,0] == self.elem[j,2]:
                    temp.append(j)
                if self.edgesHor[i,0] == self.elem[j,3] and self.edgesHor[i,1] == self.elem[j,2]:
                    temp.append(j)
                if self.edgesHor[i,1] == self.elem[j,3]:
                    temp.append(j)
                    
            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesHor[i,1] == self.elem[j,0]:
                    temp.append(j)
                if self.edgesHor[i,0] == self.elem[j,0] and self.edgesHor[i,1] == self.elem[j,1]:
                    temp.append(j)
                if self.edgesHor[i,0] == self.elem[j,1]:
                    temp.append(j)    
            self.listEdgesElements.append(temp)
        
        for i in range(0,len(self.edgesVer)):
            temp = []
            for j in range(0,len(self.elem)):
                if self.edgesVer[i,0] == self.elem[j,2]:
                    temp.append(j)
                if self.edgesVer[i,0] == self.elem[j,3]:
                    temp.append(j)
                if self.edgesVer[i,0] == self.elem[j,0] and self.edgesVer[i,1] == self.elem[j,3]:
                    temp.append(j)
                if self.edgesVer[i,1] == self.elem[j,0]:
                    temp.append(j)

            for j in range(len(self.elem)-1,-1,-1):
                if self.edgesVer[i,1] == self.elem[j,1]:
                    temp.append(j)
                if self.edgesVer[i,1] == self.elem[j,2] and self.edgesVer[i,0] == self.elem[j,1]:
                    temp.append(j)
            self.listEdgesElements.append(temp)
        return self.listEdgesElements
    
    
    # Lists Elements
    def generateListElementsEdges(self):
        """
        --- Creates the lists of every edges surrounding the elements, for every elements
        A.generateListElementsEdges(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listElementsEdges      : list
            - first term is first element, last term is last element
            - rotation rule start: left
        """
        self.listElementsEdges = []
        for i in range(0,len(self.elem)):
            temp = []
            for j in range(0,len(self.edges)):
                if self.edges[j,0] == self.elem[i,0] and self.edges[j,1] == self.elem[i,3]:
                    temp.append(j)
            for j in range(0,len(self.edges)):
                if self.edges[j,0] == self.elem[i,0] and self.edges[j,1] == self.elem[i,1]:
                    temp.append(j)
                if self.edges[j,0] == self.elem[i,1] and self.edges[j,1] == self.elem[i,2]:
                    temp.append(j)
            for j in range(0,len(self.edges)):
                if self.edges[j,1] == self.elem[i,2] and self.edges[j,0] == self.elem[i,3]:
                    temp.append(j)
            self.listElementsEdges.append(temp)
        self.listElementsEdges = np.array(self.listElementsEdges)
        return self.listElementsEdges

    def generateListElementsElements(self):
        """
        --- Creates the lists of every elements surrounding the elements, for every elements
        A.generateListElementsElements(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output                   
        self.listElementsElements      : list
            - first term is first element, last term is last element
            - rotation rule start: left
        """
        self.listElementsElements = []
        for i in range(0,len(self.elem)):
            temp = []
            for j in range(0,len(self.elem)):
                if self.elem[i,0] == self.elem[j,1] and self.elem[i,3] == self.elem[j,2]:
                    temp.append(j)
            for j in range(0,len(self.elem)):
                if self.elem[i,0] == self.elem[j,2]:
                    temp.append(j)
                if self.elem[i,0] == self.elem[j,3] and self.elem[i,1] == self.elem[j,2]:
                    temp.append(j)
                if self.elem[i,1] == self.elem[j,3]:
                    temp.append(j)
                if self.elem[i,1] == self.elem[j,0] and self.elem[i,2] == self.elem[j,3]:
                    temp.append(j)
                if self.elem[i,2] == self.elem[j,0]:
                    temp.append(j)
            for j in range(len(self.elem)-1,-1,-1):
                if self.elem[i,2] == self.elem[j,1] and self.elem[i,3] == self.elem[j,0]:
                    temp.append(j)                    
                if self.elem[i,3] == self.elem[j,1]:
                    temp.append(j)
            self.listElementsElements.append(temp)
        return self.listElementsElements

    # List main
    def listAround(self,forEach=None,get=None):
        """
        --- Gives a list of nodes, elements or edges next to nodes, elements or edges
            They are listed by following the anti-clockwise:
                example:
                    
                    2---3
                    | 0 |   ===> listElementsNodes = [[0,1,3,2]]
                    0---1
                  
                rule for the order in the list: first is left object, 
                or bottom left object if no direct object at left
                    
                  ↙ ←----←----←------←-------←----- ↖
                ↙                                     ↖
               ↓    top left     top    top right      ↑
                                o---o                  |
               start  left      |   |   right          |
               |                o---o                  ↑
               ↓  bottom left   bottom  bottom right   |
                ↘  alt start                         ↗
                  ↘ -→--------→---------→--------→ ↗
                    
        A.listAround(forEach=,get=)
        
        -- Input                will give an array (L,l) with every chosen 'get' around that nodes
        forEach         : str       
            -'Nodes'      L = N
            -'Elements'   L = N_el
            -'Edges'      L = N_ed
        get             : str
            -'Nodes'      
            -'Elements'   l depends on the combinations made between 'forEach' and 'get'
            -'Edges'      
        OR
        give='Nodes',get=None   will give an array (N,2) of every coordinates of the Nodes
        OR
        give='Edges',get='Type' will give a list (N_ed,1) of the type of the edge
        Simply:
            if the edge is on a side (top, bottom, left, right) then its 
            position (top, bottom, left, right) is written, else if it is not on a side
            then 'interior' is typed
        
        -- Ouput
        listResult      : list or array (L,l)
        """
        # print(forEach)
        # print(get)
        listResult = None
        if forEach == 'Nodes':
            if get == 'Nodes': listResult = self.generateListNodesNodes()
            elif get == 'Edges': listResult = self.generateListNodesEdges()
            elif get == 'Elements': listResult = self.generateListNodesElements()
            elif get == None: listResult = self.X_ref
            
        elif forEach == 'Edges': 
            if get == 'Nodes': listResult = self.edges
            elif get == 'Edges': listResult = None
            elif get == 'Elements': listResult = self.generateListEdgesElements()
            elif get == 'Type': listResult = self.listTypeEdges
            
        elif forEach == 'Elements':
            if get == 'Nodes': listResult = self.elem
            elif get == 'Edges': listResult = self.generateListElementsEdges()
            elif get == 'Elements': listResult = self.generateListElementsElements()
        return listResult


# =============================================================================
# Class MeshPhys for physical mesh
# =============================================================================
class MeshPhys(gF.ClassBasic):
    """
    --- Creates the physical mesh by using a reference mesh and the Transfinite Interpolation 2D
                top
            B ----------  D
            |             |
       left |             | right
            |             |
            A ----------- C
                bottom
    
    MP = MeshPhys(input1=,input2=)
    """
    # Corners A B C D values
    @property
    def A(self): return self.Corner[0,:].reshape(1,-1)
    @property
    def B(self): return self.Corner[1,:].reshape(1,-1)
    @property
    def C(self): return self.Corner[2,:].reshape(1,-1)
    @property
    def D(self): return self.Corner[3,:].reshape(1,-1)

    
    # Exports if needed
    def export_corners(self):
        return self.Corner
    
    
    # Initialization
    def __init__(self,mapTFI_bottom=None,mapTFI_top=None,mapTFI_left=None,mapTFI_right=None):
        """
        --- Initialization of the class
        MP.MeshPhys(input))

        -- Input
        All are optionnal but at least top + bottom or left + right need to be given
        A straight line will be drown for borders not given if at least one of 
        these combination were given because every corners will be known.
        The inputs are functions which follows this principle:
            def f(x):
                return function(x)
        or an equivalent
        
        Smooth functions of the sides:
        mapTFI_bottom               : A->C or mapTFI(xi,0)
        mapTFI_left                 : A->B or mapTFI(0,eta). mapTFI(0)=A, mapTFI(eta=1)=B
        mapTFI_top                  : B->D or mapTFI(xi,1)
        mapTFI_right                : C->D or mapTFI(1,eta). mapTFI(0)=C, mapTFI(eta=1)=D
        """
        # Data
        self.mapTFI_x0 = mapTFI_bottom
        self.mapTFI_0e = mapTFI_left
        self.mapTFI_x1 = mapTFI_top
        self.mapTFI_1e = mapTFI_right
        
        # Set values of corners
        self.Corner = np.zeros((4,2))           # A,B,C,D x,y coordinates array
        # Corner A = mapTFI_bottom(xi=0,eta=0) or = mapTFI_left
        try: self.Corner[0,:] = self.mapTFI_x0([0])
        except: self.Corner[0,:] = self.mapTFI_0e([0])

        # Corner B 
        try: self.Corner[1,:] = self.mapTFI_x1([0])
        except: self.Corner[1,:] = self.mapTFI_0e([1])

        # Corner C
        try: self.Corner[2,:] = self.mapTFI_x0([1])
        except: self.Corner[2,:] = self.mapTFI_1e([0])

        # Corner D
        try: self.Corner[3,:] = self.mapTFI_x1([1])
        except: self.Corner[3,:] = self.mapTFI_1e([1])
        
        # Regularization: if only 2BCs or 3BCs are given
        if mapTFI_bottom is None:
            self.mapTFI_x0 = lambda x: np.dot(1-x.reshape(-1,1),self.A) + np.dot(x.reshape(-1,1),self.C)
        if mapTFI_left is None:
            self.mapTFI_0e = lambda x: np.dot(1-x.reshape(-1,1),self.A) + np.dot(x.reshape(-1,1),self.B)
        if mapTFI_top is None:
            self.mapTFI_x1 = lambda x: np.dot(1-x.reshape(-1,1),self.B) + np.dot(x.reshape(-1,1),self.D)
        if mapTFI_right is None:
            self.mapTFI_1e = lambda x: np.dot(1-x.reshape(-1,1),self.C) + np.dot(x.reshape(-1,1),self.D)
        return

    # Max distance of borders for x and y
    @property
    def dx(self):
        """Max between BD, AC along x"""
        return max(self.D[0][0]-self.B[0][0],self.C[0][0]-self.A[0][0])
    @property
    def dy(self):
        """Max between AB, CD along y"""
        return max(self.B[0][1]-self.A[0][1],self.D[0][1]-self.C[0][1])


    # Ploting borders
    def borderPlot(self, nx=100, ny=100, s=10, lw=2,
                   show=True, init=True,
                   corners=True, borders=True,
                   legend=True):
        """ 
        --- Plot the borders AB BD DC AC with different colors 
        and the corners A B C D + their names
        B.plot()
        
        -- Input
        nx          : int       number of points of the border
        show        : boolean   show the plot or keep it active ?
        init        : boolean   initialize the plot ?
        """
        # Initialization
        fontsize=15;e=[0.05*self.dx,0.05*self.dy]
        colorBC='grey';
        if init: gF.prePlot(equalAxis=True)
        
        # Corners
        if corners:
            for k in ['A','B','C','D']:
                x,y=getattr(self,k)[0]
                plt.scatter(x,y,s=s,c='black',zorder=10)
                plt.annotate(k, (x+e[0],y+e[1]), fontsize=fontsize, zorder=100,weight='bold')
        
        # Borders (BCs)
        if borders:
            xx=np.linspace(0,1,nx)
            xy=np.linspace(0,1,ny)
            f_b = self.mapTFI_x0(xx)
            f_t = self.mapTFI_x1(xx)
            f_l = self.mapTFI_0e(xy)
            f_r = self.mapTFI_1e(xy)
            plt.plot(f_b[:,0],f_b[:,1],label='Bottom',lw=lw)
            plt.plot(f_t[:,0],f_t[:,1],label='Top',lw=lw)
            plt.plot(f_l[:,0],f_l[:,1],label='Left',lw=lw)
            plt.plot(f_r[:,0],f_r[:,1],label='Right',lw=lw)
        
        # End 
        if legend is True and borders is True: plt.legend()
        if show: plt.show()
        return 
    
    
    # Determination of each physical mesh coordinates by transfinite interpolation 2D
    """
    (x,y) = mapTFI(xi,eta)
    (x,y) = (1-xi) * mapTFI(0,eta) + xi * mapTFI(1,eta) + (1-eta) * mapTFI(xi,0)
            + eta * mapTFI(xi,1) - (1-xi) * (1-eta) * mapTFI(0,0)
            - (1-xi) * eta * mapTFI(0,1) - xi * (1-eta) * mapTFI(1,0)
            - xi * eta * r(1,1)
    As a reminder:
        mapTFI_bottom               : A->C or mapTFI(xi,0)
        mapTFI_left                 : A->B or mapTFI(0,eta). mapTFI(0)=A, mapTFI(eta=1)=B
        mapTFI_top                  : B->D or mapTFI(xi,1)
        mapTFI_right                : C->D or mapTFI(1,eta). mapTFI(0)=C, mapTFI(eta=1)=D
    And:
        self.mapTFI_x0 = mapTFI_bottom
        self.mapTFI_0e = mapTFI_left
        self.mapTFI_x1 = mapTFI_top
        self.mapTFI_1e = mapTFI_right
    Thus:
        self.mapTFI_x0 = mapTFI(xi,0)
        self.mapTFI_0e = mapTFI(0,eta)
        self.mapTFI_x1 = mapTFI(xi,1)
        self.mapTFI_1e = mapTFI(1,eta)
    """
    def meshPhys(self,XX):
        """
        --- From CPU nodes to Physical nodes
        X = A.meshPhys(x)

        -- Input
        XX           : (N,2) array      the coordinate array of CPU mesh
        xi=XX[:,0], eta=XX[:,1]
        
        -- Output
        result       : (N,2) array      result of the computation
        x=result[:,0], y=result[:,1]
        """
        xi=XX[:,0].copy()
        eta=XX[:,1].copy()
        
        """
        Computation of every single physical mesh points
        """
        # P_xi
        """Projector eta -- float Version"""
        A = np.dot(np.diag(1-xi),self.mapTFI_0e(eta)) + np.dot(np.diag(xi),self.mapTFI_1e(eta))
        
        # P_eta
        """Projector xi -- float Version"""
        B = np.dot(np.diag(1-eta),self.mapTFI_x0(xi)) + np.dot(np.diag(eta),self.mapTFI_x1(xi))
        
        # P_xi_eta
        """P_xi P_eta: Composite -- float Version"""
        z = np.zeros(len(eta))
        o = np.ones(len(eta))
        
        Z = np.dot(np.diag(1-eta),self.mapTFI_x0(z)) + np.dot(np.diag(eta),self.mapTFI_x1(z))
        O = np.dot(np.diag(1-eta),self.mapTFI_x0(o)) + np.dot(np.diag(eta),self.mapTFI_x1(o))
        
        C = np.dot(np.diag(1-xi),Z)+np.dot(np.diag(xi),O)
        return A + B - C


    # Physical mesh building
    def buildMesh(self,mapTFI=None,
                  X_ref=None,useCPU=True,n=None,m=None):
        """ 
        --- Build Physical Mesh from Mapping and CPU Mesh
        C.buildMesh()
        
        -- Inputs
        mapTFI          : transfinite interpolation 2D      see examples and notebook to 
                                                            understand what it is
        X_ref           : array (N,2)       coordinates of the CPU mesh
                                            can also be just random coordinates 
                                            you chose between 0 and 1
                                            (default = None)
        useCPU          : boolean           override X_ref input to use the MeshCPU class to
                                            create the X_ref points coordinates and use them for 
                                            physical mesh depending on n and m
                                            (default = True)
            
        -- Output
        X               : array (N,2)       coordinates of physical mesh
        """
        if mapTFI is None:
            mapTFI = self.meshPhys
            
        if useCPU == True:  
            try: 
                CPU = MeshCPU(n=n,m=m)
                X_ref = CPU.export_X_ref()
            except: print("Please define n and m for physical mesh building")
            
        try: 
            X = mapTFI(X_ref)
            return X
        except: print('Mapping self.mapTFI not defined. Physical Mesh not built.')
        return 


# =============================================================================
# Class Mesh for meshing functions
# =============================================================================
class MeshPlot(gF.ClassBasic,Properties):
    """
    Functions to use on nodes arrays
    """
    def __init__(self, X, n, m,
                 show=True,
                 nodes=True, nodesNames=True,
                 elementsNames=True,
                 edges = True, edgesNames=True,
                 size_x=10, size_y=10):
        """
        --- Gives a representation of the mesh with nodes and segments
        Mesh(X=, n=, m=)
        
        -- Input
        X               : array (N,2)       Coordinates of nodes to plot
        n,m         : int           meshing precision properties
        elem            : array (N_el,4)    Nodes that compose every elements   
                                            (default:None but becomes MeshCPU.export_elem() if needed)
        edges           : array (N_ed,2)    Nodes that compose every edges
                                            (default:None but becomes MeshCPU.export_edges() if needed)
        size_x          : float             Physical horizontal size of plot    (default: 5) (better not touch)
        size_y          : float             Physical vertical size of plot      (default: 5) (better not touch)
        show            : boolean           Show plot or keep active ?          (defaut: True)
        nodes           : boolean           Draw physical Nodes on plot ?       (default: True)
        nodesNames      : boolean           Show Nodes names on plot ?          (default: True)
        elementsNames   : boolean           Show elements names on plot ?       (default: True)
        edgesHorizontal : boolean           Draw horizontal edges on plot ?     (default: True)
        edgesVertical   : boolean           Draw vertical edges on plot ?       (default: True)
        edgesNames      : boolean           Show edges names on plot ?          (default: True)
        
        -- Output
        A plot in console or plot pane with the points given and depending on the options chosen
        """
        # Data
        self.n = n
        self.m = m
        self.X = X
        
        # Setting up plot
        plt.rcParams["figure.figsize"] = (size_x, size_y)

        # Setting elem and edges
        CPU = MeshCPU(n=self.n,m=self.m)
        self.edges = CPU.export_edges()
        self.elem = CPU.export_elem()
        
        # Drawing 
        if edges: self.plot_edges()
        if nodes: self.plot_nodes()
        if nodesNames: self.plot_nodes_names()
        if elementsNames: self.plot_elements_names()
        if edgesNames: self.plot_edges_names()
        
        # End
        if show: plt.show()
        return


    # Plot nodes, elements and edges
    def plot_edges(self):
        """ Draws the edges between points of the mesh """
        for i in range(0,self.N_ed):
            x = (self.X[self.edges[i,0],0],self.X[self.edges[i,1],0])
            y = (self.X[self.edges[i,0],1],self.X[self.edges[i,1],1])
            plt.plot(x,y,c='black',zorder=1)
        
    def plot_nodes(self):
        """ Draws the nodes of mesh """
        plt.plot(self.X[:,0],self.X[:,1],marker='+',c='r',zorder=2,linestyle='None')
        return
        
    def plot_nodes_names(self):
        """ Types the numbers of the Nodes on the plot """
        for i in range(0,self.N):
            plt.annotate(i, (self.X[i,0], self.X[i,1]))
        return
    
    def plot_elements_names(self):
        """ 
        Types the numbers of the elements on the plot 
        REQUIRES CPU.elem
        """
        for i in range(0,self.N_el):
            coordEl = self.X[self.elem[i]]
            plt.annotate(i, ((coordEl[0,0] + coordEl[1,0]+coordEl[2,0]+coordEl[3,0])/4, (coordEl[0,1] + coordEl[1,1] + coordEl[2,1] + coordEl[3,1])/4))
        return
    
    def plot_edges_names(self):
        """ 
        Types the numbers of the edges on the plot 
        REQUIRES CPU.edges
        """
        for i in range(0,self.N_ed):
            coordEl = self.X[self.edges[i]]
            plt.annotate(i, ((coordEl[0,0] + coordEl[1,0])*1/2, (coordEl[0,1] + coordEl[1,1])*1/2)) 
        return
    
    

# =============================================================================
# Class Cubic Splines for 1D Interpolation
# =============================================================================
class CubicSplines(gF.ClassBasic):
    """
    A = CubicSplines(x=,y=,BC_left='extrapolated',ddy_0=0, ddy_n=0, BC_right='derivative',dy_0=None,dy_n=1.2, build=True) -- Builds Splines
    or 
    A = CubicSplines(x=,y=,BC_left='continuous',ddy_0=0, ddy_n=0, BC_right='prescribed',dy_0=None,dy_n=None); A.build()
    fx   = A(xx) or fx=A.f(xx)
    dfx  = A.df(xx)
    d2fx = A.ddf(xx)

    -- BC Types - left : ddy_0, dy_0 (y'', y' respectively) 
                - right: ddy_n, dy_n (y'', y' respectively) 
    0 or 'prescribed'       : y''(x0)=ddy_0 (given by user) or y''(xn)=ddy_n
    1 or 'continuous'       : y''(x0)=y''(x1)               or y''(xn)=y''(xn-1) --> More Curvature at end points
    2 or 'extrapolated'     : y''(x0)=Linear Extrapolation from y''(x1),y''(x2), and similarly for y''(xn)
    3 or 'derivative'       : y'(x0) is fixed by user (resp. y'(xn)), by providing dy_0 or dy_n

    -- CODE VALIDATED! Works well
    """
    def __init__(self,x=None,y=None,build=True,
                 ddy_0=0, ddy_n=0, dy_0=None, dy_n=None,
                 BC_left='extrapolated', BC_right='extrapolated'):
        """
        -- Input
        x              : (n+1,) x-values
        y              : (n+1,) y-values (function)
        ddy_0,ddy_n    : floats - default values for y''(x0), y''(xn). Used if BC_=0 or 'prescribed'
        dy_
        """
        # Data
        self.x = x.copy()
        self.y = y.copy()
        self.ddy_0 = ddy_0
        self.ddy_n = ddy_n
        self.dy_0 = dy_0
        self.dy_n = dy_n
        self.BC_left = BC_left
        self.BC_right = BC_right

        # Sanity Check
        assert self.BC_left in [0,1,2,3,'prescribed','continuous','extrapolated','derivative']
        assert self.BC_right in [0,1,2,3,'prescribed','continuous','extrapolated','derivative']
        assert len(self.x)==len(self.y)

        # Vector t
        self.t = np.hstack([x[i+1]-x[i] for i in range(self.n)])
        
        # Build
        if build: self.build()
        return

    @property
    def n(self): return len(self.x)-1


    def initMatrix(self):
        """
        Initialize Matrix A, vector b
        """
        # Data
        n = self.n; t = self.t; y = self.y

        # -- Matrix A
        A = np.zeros((n-1,n-1))

        # Diagonal, Sup, Inf
        for i in range(n-1): A[i,i] = 2*t[i]+2*t[i+1]
        for i in range(n-2): A[i,i+1]=t[i+1]
        for i in range(n-2): A[i+1,i]=t[i+1]
        self.A = A.copy()
            
        # -- Vector b
        b = np.zeros(n-1)
        for i in range(n-1):
            b[i] = (y[i+2]-y[i+1])/t[i+1]-(y[i+1]-y[i])/t[i]
        self.b = 6*b
        return 

    def _setMatrixBC(self):
        """
        Update BC for Matrix A, vector b
        """
        # ---- Set BC y''(0)
        # Prescribed
        if self.BC_left in [0,'prescribed']:
            self.b[0] = self.b[0]-self.ddy_0*self.t[0]

        # Continuous: y''(0)=y''(1)
        if self.BC_left in [1,'continuous']:
            self.A[0][0] += self.t[0]

        # Extrapolated from y''1,y''2
        if self.BC_left in [2,'extrapolated']:
            self.A[0][0] += self.t[0]*(self.t[0]+self.t[1])/self.t[1]
            self.A[0][1] += -self.t[0]*self.t[0]/self.t[1]

        # y'(0) prescribed
        if self.BC_left in [3,'derivative']:
            assert self.dy_0 is not None
            self.A[0][0] += -self.t[0]*0.5
            self.b[0] += 3*self.dy_0-3*(self.y[1]-self.y[0])/self.t[0]


        # ---- Set BC y''(n)
        # Prescribed
        if self.BC_right in [0,'prescribed']:
            self.b[-1] += -self.ddy_n*self.t[-1]

        # Continuous: y''(n)=y''(n-1)
        if self.BC_right in [1,'continuous']:
            self.A[-1][-1] += self.t[-1]

        # Extrapolated from y''n-1,y''n-2
        if self.BC_right in [2,'extrapolated']:
            self.A[-1][-1] += self.t[-1]*(self.t[-2]+self.t[-1])/self.t[-2]
            self.A[-1][-2] += -self.t[-1]*self.t[-1]/self.t[-2]

        # y'(n) prescribed
        if self.BC_right in [3,'derivative']:
            assert self.dy_n is not None
            self.A[-1][-1] += -self.t[-1]*0.5
            self.b[-1] += -3*self.dy_n+3*(self.y[-1]-self.y[-2])/self.t[-1]
            
        return 

    def _setSecondDerivativeBC(self):
        """
        Update y''(0), y''(n)
        """
        # ---- Set BC y''(0)
        # Prescribed
        if self.BC_left in [0,'prescribed']:
            self.d2y[0] = self.ddy_0

        # Continuous: y''(0)=y''(1)
        if self.BC_left in [1,'continuous']:
            self.d2y[0] = self.d2y[1]

        # Extrapolated from y''1,y''2
        if self.BC_left in [2,'extrapolated']:
            self.d2y[0] = self.d2y[1]*(self.t[0]+self.t[1])/self.t[1] - self.d2y[2]*self.t[0]/self.t[1]

        # y'(0) prescribed
        if self.BC_left in [3,'derivative']:
            assert self.dy_0 is not None
            self.d2y[0] = -0.5*self.d2y[1] + 3*(self.y[1]-self.y[0])/self.t[0]/self.t[0] - 3*self.dy_0/self.t[0]

        # ---- Set BC y''(n)
        # Prescribed
        if self.BC_right in [0,'prescribed']:
            self.d2y[-1] = self.ddy_n

        # Continuous: y''(n)=y''(n-1)
        if self.BC_right in [1,'continuous']:
            self.d2y[-1] = self.d2y[-2]

        # Extrapolated from y''1,y''2
        if self.BC_right in [2,'extrapolated']:
            self.d2y[-1] = self.d2y[-2]*(self.t[-1]+self.t[-2])/self.t[-2] - self.d2y[-3]*self.t[-1]/self.t[-2]

        # y'(0) prescribed
        if self.BC_right in [3,'derivative']:
            assert self.dy_n is not None
            self.d2y[-1] = -0.5*self.d2y[-2] - 3*(self.y[-1]-self.y[-2])/self.t[-1]/self.t[-1] + 3*self.dy_n/self.t[-1]
            
        return
    
    def build(self):
        """
        Evaluation of y''(xi)
        """
        # Initialize Matrix, b
        self.initMatrix()

        # Add BC
        self._setMatrixBC()

        # Solve For y''
        self.d2y = np.zeros(self.n+1)
        self.d2y[1:-1] = np.linalg.solve(self.A,self.b)

        # Update d2y left/right
        self._setSecondDerivativeBC()
        
        return

    def phi_i(self,xx,i=0):
        """
        Basis Function Cubic Spline, depending on y''
        -- phi_(i+1)(x), i in [0,n-1]
        """
        t=self.t[i]
        return self.y[i]*(self.x[i+1]-xx)/t + self.y[i+1]*(xx-self.x[i])/t + \
            1.0/6.0*self.d2y[i]*( (self.x[i+1]-xx)**3/t - t*(self.x[i+1]-xx)) + \
            1.0/6.0*self.d2y[i+1]*( (xx-self.x[i])**3/t - t*(xx-self.x[i]))

    def d_phi_i(self,xx,i=0):
        """
        Derivative Basis Function Cubic Spline, depending on y''
        -- d_phi_(i+1)(x), i in [0,n-1]
        """
        t=self.t[i]
        return  (self.y[i+1]-self.y[i])/t + 1.0/6.0*t*(self.d2y[i]-self.d2y[i+1]) + \
            0.5*self.d2y[i+1]/t*(xx-self.x[i])**2-\
            0.5*self.d2y[i]/t*(self.x[i+1]-xx)**2 

    def d2_phi_i(self,xx,i=0):
        """
        Derivative Second Basis Function Cubic Spline, depending on y''
        -- d2_phi_(i+1)(x), i in [0,n-1]
        """
        t=self.t[i]
        return self.d2y[i+1]/t*(xx-self.x[i]) + self.d2y[i]/t*(self.x[i+1]-xx)

    def f(self,xx):
        """
        Function Evaluation - xx=(n,) array
        """
        # List of indices: Remove 0 values to 1, i=ind-1. Then apply phi_{i+1}(xx)
        ind  = self.x.searchsorted(xx)
        ind[ind==0]=1
        i = ind-1

        # Evaluate
        fx = np.hstack( [self.phi_i(xx[k],i=i[k]) for k in range(len(xx))])
        return fx

    def df(self,xx):
        """
        Derivative Evaluation
        """
        # List of indices: Remove 0 values to 1, i=ind-1. Then apply phi_{i+1}(xx)
        ind  = self.x.searchsorted(xx)
        ind[ind==0]=1
        i = ind-1

        # Evaluate
        fx = np.hstack( [self.d_phi_i(xx[k],i=i[k]) for k in range(len(xx))])

        return fx

    def ddf(self,xx):
        """
        Second Derivative Evaluation
        """
        # List of indices: Remove 0 values to 1, i=ind-1. Then apply phi_{i+1}(xx)
        ind  = self.x.searchsorted(xx)
        ind[ind==0]=1
        i = ind-1

        # Evaluate
        fx = np.hstack( [self.d2_phi_i(xx[k],i=i[k]) for k in range(len(xx))])
        return fx

# =============================================================================
# Class Cubic Splines for (1D)-(d-D) Interpolation
# =============================================================================
class CubicSplines_d(gF.ClassBasic):
    """

    """
    @property
    def n(self): return self.M.shape[0]

    @property
    def d(self): return self.M.shape[1]
    
    def __init__(self, M=None,Points=[],**opt):
        """
        M            : (N,d) array, d=2 or 3. 
        or Points    : List of (d,) array
        **opt        : Options For Cubic Splines
        M[0]=A (xi=0), M[-1]=B (xi=1); intermediate points are sorted accordingly
        """
        if len(Points) !=0:
            M = np.vstack(Points)
        self.M = M.copy()

        # CumSum for Curvilinear Abscissa
        self.s = gF.normCumDist(M)
        self.opt = deepcopy(opt)

        # Build Interpolator Cubic Splines
        self.build()
        return

    def build(self):
        """
        (s,x), (s,y) and (s,z) with Cubic Splines
        """
        self.CS = {}
        for k in range(self.d):
            self.CS[k] = CubicSplines(x=self.s,y=self.M[:,k],build=True,**self.opt)
        return

    def f(self,xi):
        """
        Mxi = self.f(xi), Evaluation Points for xi€[0,1]

        -- Input
        xi                : (n,) array in [0,1]
        
        -- Output
        Mxi               : (n,d) array
        """
        n = len(xi)
        S = np.zeros((n,self.d))
        for k in range(self.d):
            S[:,k] = self.CS[k].f(xi)
        return S

    def tangent(self,xi):
        """(n,)->(n,d) Tangent Vector, of norm 1"""
        return self.df(xi,normalized=True)

    def normal(self,xi):
        """(n,)->(n,d) Normal Vector, of norm 1"""
        t = self.tangent(xi)
        n = np.zeros(t.shape)
        n[:,0]=-t[:,1]
        n[:,1]=t[:,0]
        return n
    
    def df(self,xi,normalized=False):
        """
        Mxi = self.df(xi), Evaluation Derivative for xi€[0,1]
        if normalized, each derivative norm=1
        -- Input
        xi                : (n,) array in [0,1]
        
        -- Output
        Mxi               : (n,d) array
        """
        n = len(xi)
        S = np.zeros((n,self.d))
        for k in range(self.d):
            S[:,k] = self.CS[k].df(xi)

        if normalized:
            for i in range(n): 
                S[i,:]=S[i,:]/np.linalg.norm(S[i,:])
        return S

    def ddf(self,xi):
        """
        Mxi = self.ddf(xi), Evaluation Second Derivative for xi€[0,1]

        -- Input
        xi                : (n,) array in [0,1]
        
        -- Output
        Mxi               : (n,d) array
        """
        n = len(xi)
        S = np.zeros((n,self.d))
        for k in range(self.d):
            S[:,k] = self.CS[k].ddf(xi)
        return S
    
    
    
# =============================================================================
# Export to file and Import from file (IMPORT IS OBSOLETE)
# =============================================================================
class Export(Properties):
    # Initialization
    def __init__(self,X=None,n=None,m=None):
        self.X = X
        self.n = n
        self.m = m
        
        CPU = MeshCPU(n=n,m=m)
        self.elem = CPU.export_elem()
        self.edges = CPU.export_edges()
        return
    
    # Export to file
    def exportFile(self,format,filename='mesh'):
        """
        --- Creates a filename.format file and writes the results of A.generateNodes() on it
        A.exportFile(format=,filename=)
        or
        A.exportFile(format=)
        
        -- Input
        format          : str       format to export to
            - 'su2'
            - 'msh'
            - 'vtk'
        filename        : str       name of the  file (optionnal)              (default: 'meshRef')
        build           : boolean   generate the nodes again ? (optional)      (default: False)
        """
        path = path_main + "/mesh"
        os.chdir(path)
        if format == 'su2': self.exportToSU2(filename)
        elif format == 'msh': self.exportToMSH(filename)
        elif format == 'vtk': self.exportToVTK(filename)
        else: print("Error in format")
        os.chdir(path_main+"/src")
        return
            
    def exportToSU2(self,filename='meshRef'):
        """
        --- Creates a filename.su2 file and writes the results of A.generateNodes() on it
        A.exportToSU2()
        or
        A.exportToSU2(filename)
        
        -- Input
        filename        : str       name of the  file
        """
        # Initialization: Create or replace filename and write dimension
        filename += ".su2"
        su2_file = open(filename,'w+')
        su2_file.write("% Mesh dimension\n")
        su2_file.write("NDIME= 2\n")
        su2_file.write("\n")
        
        # Nodes input
        su2_file.write("% Nodes coordinates\n")
        su2_file.write("NPOIN= {}\n".format(self.N))
        for i in range(0,self.N):
            su2_file.write("{X:.16f} {Y:.16f} {k}\n".format(X=self.X[i,0],Y=self.X[i,1],k=i))
        
        #Elements input
        su2_file.write("\n% Element list\n")
        su2_file.write("NELEM= {}\n".format(self.N_el))
        for i in range(0,self.N_el):
            su2_file.write("4 {N1} {N2} {N3} {N4}\n".format(N1=self.elem[i,0],N2=self.elem[i,1],N3=self.elem[i,2],N4=self.elem[i,3]))
        
        #Limits (Markers)
        su2_file.write("\n% Markers list\n")
        su2_file.write("NMARK= 4\n")
        
        nMarker_lower = self.n - 1
        nMarker_side = self.m - 1
        
        su2_file.write("MARKER_TAG= lower\n")
        su2_file.write("MARKER_ELEMS= {}\n".format(nMarker_lower))
        for i in range(0,self.n-1):
            su2_file.write("{type} {N1} {N2}\n".format(type=3,N1=i,N2=i+1))
        
        su2_file.write("MARKER_TAG= upper\n")
        su2_file.write("MARKER_ELEMS= {}\n".format(nMarker_lower))
        for i in range(0,self.n-1):
            su2_file.write("3 {N1} {N2}\n".format(N1=(self.m-1)*self.n + i,N2=(self.m-1)*self.n + i+1))
        
        su2_file.write("MARKER_TAG= right\n")
        su2_file.write("MARKER_ELEMS= {}\n".format(nMarker_side))
        for i in range(0,self.m-1):
            su2_file.write("3 {N1} {N2}\n".format(N1=self.n -1 + (self.m+1)*i,N2=self.n -1 + (self.m+1)*(i+1)))
        
        su2_file.write("MARKER_TAG= left\n")
        su2_file.write("MARKER_ELEMS= {}\n".format(nMarker_side))
        for i in range(0,self.m-1):
            su2_file.write("3 {N1} {N2}\n".format(N1= + (self.m+1)*i,N2= + (self.m+1)*(i+1)))
        
        su2_file.close()
        return
        
    def exportToMSH(self,filename='meshRef'):
        """
        --- Creates a filename.msh file and writes the results of A.generateNodes() on it
        A.exportToMSH()
        or
        A.exportToMSH(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        """
        # Initialization: Create or replace filename and write dimension
        filename += ".msh"
        msh_file = open(filename,'w+')
        msh_file.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        
        # Nodes input
        msh_file.write("$Nodes\n")
        msh_file.write("{}\n".format(self.N))
        for i in range(0,self.N):
            msh_file.write("{k} {X:.16f} {Y:.16f} {Z:.16f}\n".format(X=self.X[i,0],Y=self.X[i,1],k=i+1,Z=0))
        msh_file.write("$EndNodes\n")
        
        #Elements input
        msh_file.write("$Elements\n")
        msh_file.write("{}\n".format(self.N_el))
        for i in range(0,self.N_el):
            msh_file.write("{K}  3  2  0  {K}  {N1}  {N2}  {N3}  {N4}\n".format(K=i+1,N1=self.elem[i,0]+1,N2=self.elem[i,1]+1,N3=self.elem[i,2]+1,N4=self.elem[i,3]+1))
        msh_file.write("$EndElements\n")
        msh_file.close()
        return

    def exportToVTK(self,filename='meshRef'):
        """
        --- Creates a filename.vtk file and writes the results of A.generateNodes() on it
        A.exportToVTK()
        or
        A.exportToVTK(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        """
        # Initialization: Create or replace filename and write dimension
        filename += ".vtk"
        vtk_file = open(filename,'w+')
        vtk_file.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        
        # Nodes (Points) input
        vtk_file.write("POINTS {} float\n".format(self.N))
        for i in range(0,self.N):
            vtk_file.write("{X:.16f} {Y:.16f} {Z:.16f}\n".format(X=self.X[i,0],Y=self.X[i,1],Z=0))
        vtk_file.write("\n")

        #Elements (Cells) input
        vtk_file.write("CELLS {N_el} {size} \n".format(N_el=self.N_el,size=self.N_el*5))
        for i in range(0,self.N_el):
            vtk_file.write("4 {N1} {N2} {N3} {N4}\n".format(N1=self.elem[i,0],N2=self.elem[i,1],N3=self.elem[i,2],N4=self.elem[i,3]))
        vtk_file.write("\n")
        
        #Elements (Cell_Type) input
        vtk_file.write("CELL_TYPES {N_el}\n".format(N_el=self.N_el))
        for i in range(0,self.N_el):
            vtk_file.write("9\n")
        vtk_file.close()
        return
        
    
class Import():
    #############################################################################################
    # OBSOLETE
    # IMPORTANT TO NOTE THAT THIS ONLY GIVES n AND m SO IT ACTUALLY DOESN'T WORK ON PHYSICAL MESH
    # AND DOESN'T SEND BACK X, elem, edges.
    # IT ONLY NEEDS TO BE REWRITEN A BIT
    #############################################################################################
    
    # Import from file
    def importFile(self,format,filename='meshRef'):
        """
        --- Creates a filename.format file and send the results of self.n and self.m
        A.importFile(format=,filename=)
        or
        A.importFile(format=)
        
        -- Input
        format          : str       format of the the file to import from
            - 'su2'
            - 'msh'
            - 'vtk'
        filename        : str       name of the  file (optionnal)
        """
        if format == 'su2': self.importFromSU2(filename)
        if format == 'msh': self.importFromMSH(filename)
        if format == 'vtk': self.importFromVTK(filename)
        return self.n, self.m
    
    def importFromSU2(self,filename='meshRef'):
        """
        --- Reads a filename.su2 file and sends back the results
        A.importFromSU2()
        or
        A.importFromSU2(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        
        -- Ouput
        X_ref           : (3,n)     array of coordinates of Nodes k array
        """
        # Initialization: Reads the file and clears it of newline characters (\n)
        filename += '.su2'
        su2_file = open(filename,'r')
        file = su2_file.readlines()
        for i in range(0,len(file)):
            file[i] = file[i].strip('\n')

        # Search for "MARKER_TAG= lower" : Extracts n
        for i in range(0,len(file)):
            if file[i].find("MARKER_TAG= lower") == 0:
                self.n = int(file[i+1].strip("MARKER_ELEMS= ")) + 1
                break
        
        # Search for "MARKER_TAG= right" : Extacts m
        for i in range(0,len(file)):
            if file[i].find("MARKER_TAG= right") == 0:
                self.m = int(file[i+1].strip("MARKER_ELEMS= ")) + 1
                break
        su2_file.close()
        return
        
    def importFromMSH(self,filename='meshRef'):
        """
        --- Reads a filename.msh file and writes the results of A.generateNodes() in self.X_ref
        A.importFromMSH()
        or
        A.importFromMSH(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        
        -- Ouput
        X_ref           : (3,n)     array of coordinates of Nodes k array
        """
        # Initialization: Reads the file and clears it of newline characters (\n)
        filename += '.msh'
        msh_file = open(filename,'r')
        file = msh_file.readlines()
        for i in range(0,len(file)):
            file[i] = file[i].strip('\n')

        # Search for "$Nodes" : Extracts N
        for i in range(0,len(file)):
            if file[i].find("$Nodes") == 0:
                pos = i
                N = int(file[i+1])
                break
        
        # Search for "$Elements" : Extracts N_el
        for i in range(0,len(file)):
            if file[i].find("$Elements") == 0:
                N_el = int(file[i+1])
                break
        
        # Extract X_ref to determine n
        X_ref = np.zeros((N,2))
        for i in range(pos+2,pos+2+N):
            X_ref[i-5,0] = file[i][len(str(i)):len(str(i))+16]
            X_ref[i-5,1] = file[i][len(str(i))+19:len(str(i))+19*2]

        # Extraction of n            
        for i in range(0,N):
            self.n = i+1
            if X_ref[i,0] == 1: break
        
        # Computing of m : m = N/n
        self.m = int(N/self.n)
        msh_file.close()
        return
        
    def importFromVTK(self,filename='meshRef'):
        """
        --- Reads a filename.vtk file and writes the results of A.generateNodes() in self.X_ref
        A.importFromVTK()
        or
        A.importFromVTK(filename)
        
        -- Input
        filename        : str       name of the  file (optionnal)
        
        -- Ouput
        X_ref           : (3,n)     array of coordinates of Nodes k array
        """
        # Initialization: Reads the file and clears it of newline characters (\n)
        filename += '.vtk'
        vtk_file = open(filename,'r')
        file = vtk_file.readlines()
        for i in range(0,len(file)):
            file[i] = file[i].strip('\n')
        
        # Search for "POINTS " : Extracts N
        pos = 3
        N = int(file[4][7:len(file[4])].strip('float'))
        
        # Search for "$Elements" : Extracts N_el
        for i in range(0,len(file)):
            if file[i].find("$Elements") == 0:
                N_el = int(file[i+1])
                break
        
        # Extract X_ref to determine n
        X_ref = np.zeros((N,2))
        for i in range(pos+2,pos+2+N):
            X_ref[i-5,0] = file[i][0:16]
            X_ref[i-5,1] = file[i][19:19*2]

        # Extraction of n            
        for i in range(0,N):
            self.n = i+1
            if X_ref[i,0] == 1: break

        # Computing of m : m = N/n
        self.m = int(N/self.n)
        vtk_file.close()
        return


# =============================================================================
# Functions NACA
# =============================================================================
def profileNACA_00t(x=None, t=12, chord=1):
    """
    -- (x,y) = Upper Profile NACA_00t
    For chord=1
    """
    # Correction Coefficient so sum(a)==0, so y(chord)=0
    a=np.array([0.29690,-0.1260,-0.3516,0.2843,-0.136 ])
    a[-1] = a[-1]-a.sum()

    # Evaluation
    x = x/chord
    y = 0.01*5*t*(a[0]*x**0.5+a[1]*x + a[2]*x**2+ a[3]*x**3+ a[4]*x**4)
    return y*chord

def genNACA_00t(N=10, t=12, chord=1):
    """
    X = genNACA_00t(n=,t=,chord=), X = (n,2) array
    """
    X = np.zeros((N,2))
    x = np.linspace(0,1,N)*chord
    y = profileNACA_00t(x=x, t=t, chord=chord)
    X[:,0] = x
    X[:,1] = y
    return X



# =============================================================================
# Spline interpolation on border plot
# =============================================================================
def splinePlot(X,MM,title=None,show=True,equalAxis=False):
    plt.title(title)
    plt.scatter(X[:,0],X[:,1],s=15,c='black',zorder=2)
    plt.scatter(MM[:,0],MM[:,1],s=2,c='red',zorder=3)
    if show: plt.show()
    return




# =============================================================================
# Main 
# WORKING EXAMPLES (set to True to use)
# =============================================================================
if __name__=='__main__':
    # CPU mesh and tools
    if True:
        n = 6; m = 3
        ref = MeshCPU(n,m)
        # Plot
        mesh = MeshPlot(X=ref.export_X_ref(), n=n, m=m,
                 nodes=True, nodesNames=True,
                 elementsNames=True, 
                 edges=True, edgesNames=True)
    
    
    # Edges type example
    if False:
        n = 6; m = 3
        ref = MeshCPU(n,m)
        print(ref.listAround(forEach='Edges',get='Type'))
    
    # Fonctions possible for Nodes Edges and Elements
    if False:
        # Setup
        n = 5; m = 4
        ref = MeshCPU(n,m)
        
        # Nodes
        print("Total amount of Nodes:\n{}\n".format(ref.N))
        print("Coordinates of a Node:\n{}\n".format(ref.listAround(forEach='Nodes')[0]))
        print("Coordinates of some Nodes:\n{}\n".format(ref.listAround(forEach='Nodes')[0:4]))
        
        print("Coordinates of every Nodes:\n{}\n".format(ref.listAround(forEach='Nodes')))
        
        print("Positions for every Nodes:\n{}\n".format(ref.nodePosition()))
        print("Node number for i and j:\n{}\n".format(ref.nodePositionInv(2,0)))

        # Elements
        print("Total amount of elements:\n{}\n".format(ref.N_el))
        print("List of Nodes for each elements:\n{}\n".format(ref.listAround(forEach='Elements',get='Nodes')))
        print("List of Nodes for one element:\n{}\n".format(ref.listAround(forEach='Elements',get='Nodes')[0]))
        print("Coordinates of Nodes of an element:\n{}\n".format(ref.export_X_ref()[ref.listAround(forEach='Elements',get='Nodes')[0]]))
        
        # Edges
        print("Total amount of Edges:\n{}\n".format(ref.N_ed))
        print("List of Nodes for each Edges:\n{}\n".format(ref.listAround(forEach='Edges',get='Nodes')))
        print("List of position of Edge:\n{}\n".format(ref.listAround(forEach='Edges',get='Type')))
        
        # Lists
        print("List of get around each forEach:\n{}\n".format(ref.listAround(forEach='Nodes',get='Nodes')))
        
        # Plot
        mesh = MeshPlot(X=ref.export_X_ref(), n=n, m=m,
                 nodesNames=False,
                 elementsNames=False, edgesNames=True)
        mesh = MeshPlot(X=ref.export_X_ref(), n=n, m=m,
                 nodes=False, nodesNames=False,
                 elementsNames=False, 
                 edgesNames=False)


        
    # Physical mesh example: NACA
    if True:
        """
        We need first to define the Boundary Conditions BC (borders).
        We define bottom top left right but we could define only two.
        The important part is knowing the corners, 
        this is obtained via 2 border that don't have any common corner
        bottom = NACA profile
        top = circular arc
        left, right = random circular arc shape
        """
        # bottom border
            # NACA Profile
        X = genNACA_00t(N=50,chord=5)

            # Points
        ii = np.linspace(0.01,5,100)+5
        x = np.vstack([[ii[k],0] for k in range(len(ii))])
        X = gF.concatRow([X,x])

            # Cubic Splines Interpolation
        A = CubicSplines_d(M=X.copy())
        MM = A.f(np.linspace(0,1,100))

            # mapTFI_bottom
        def mapTFI_bottom(x):
            return A.f(x)

            # Plot
        gF.prePlot(equalAxis=True)
        splinePlot(X=X,MM=MM,title='mapTFI_bottom')


        # top border
            # Circular Arc
        N = 50; r=5 ; d=2 
        X = np.zeros((N,d)); t = np.linspace(np.pi/2,np.pi,N)[::-1]
        X[:,0] = r*np.cos(t)
        X[:,1] = r*np.sin(t)

            # Points
        ii = np.linspace(0.01,10,100)
        x = np.vstack([[ii[k],5] for k in range(len(ii))])
        X = gF.concatRow([X,x])

            # Cubic Splines Interpolation
        B = CubicSplines_d(M=X.copy())
        MM = B.f(np.linspace(0,1,100))

            # mapTFI_top
        def mapTFI_top(x):
            return B.f(x)

            # Plot
        gF.prePlot(equalAxis=True)
        splinePlot(X=X,MM=MM,title='mapTFI_top')

        
        # ---- mapTFI_left
            # Circular Arc
        N = 50; r=1 ; d=2 ; p=5/4
        X = np.zeros((N,d)); t = np.linspace(np.pi+np.pi/N,2*np.pi,N)[::-1]
        X[:,0] = (r*np.cos(t)-r)*p
        X[:,1] = (r*np.sin(t)/3)*p
        
        x = np.zeros((N,d)); t = np.linspace(0,np.pi,N)[::]
        x[:,0] = ((r*np.cos(t)-r*2)-r)*p
        x[:,1] = (r*np.sin(t)/3)*p

            # Points
        X = gF.concatRow([X,x])

            # Cubic Splines Interpolation
        C = CubicSplines_d(M=X.copy())
        MM = C.f(np.linspace(0,1,100))

            # mapTFI_left
        def mapTFI_left(x):
            return C.f(x)

            # Plot
        gF.prePlot(equalAxis=True)
        splinePlot(X=X,MM=MM,title='mapTFI_left')
        
        
        # ---- mapTFI_right
            # Circular Arc
        N = 50; r=1 ; d=2 ; p=5/4
        X = np.zeros((N,d)); t = np.linspace(2*np.pi,np.pi+np.pi/N,N)[::]
        X[:,1] = (r*np.cos(t)-r)*p+5
        X[:,0] = (r*np.sin(t)/2)*p+10
        
        x = np.zeros((N,d)); t = np.linspace(np.pi,0,N)[::-1]
        x[:,1] = ((r*np.cos(t)-r*2)-r)*p+5
        x[:,0] = (r*np.sin(t)/2)*p+10
        
            # Points
        X = gF.concatRow([X,x])
        x1 = np.zeros((len(X[:,0]),2))
        for i in range(0,len(X[:,0])):
            x1[i,0] = X[len(X[:,0])-i-1,0]
            x1[i,1] = X[len(X[:,0])-i-1,1]

            # Cubic Splines Interpolation
        D = CubicSplines_d(M=x1.copy())
        MM = D.f(np.linspace(0,1,100))

            # mapTFI_right
        def mapTFI_right(x):
            return D.f(x)

            # Plot
        gF.prePlot(equalAxis=True)
        splinePlot(X=X,MM=MM,title='mapTFI_right')
        
        
        """
        Thanks to the BCs (Boundary Conditions), we can define the physical mesh
        """
        # Generate TFI Mesh
            # Setup CPU
        n = 30 ; m = 20
        ref = MeshCPU(n,m)
        
            # CPU Plot
        """
        mesh = MeshPlot(X=ref.export_X_ref(), n=n, m=m,
                 nodes=False, nodesNames=False,
                 elementsNames=False, edgesNames=False)
        """
        
            # TFI_2D and mapping
        AA = MeshPhys(mapTFI_top=mapTFI_top,mapTFI_bottom=mapTFI_bottom,mapTFI_left=mapTFI_left,mapTFI_right=mapTFI_right)

            # Physical Mesh build and plot
        X = AA.buildMesh(n=n,m=m)
        # Plot borders
        AA.borderPlot(nx=n, ny=m,
                      show=False)
        # Plot interior points and lines
        mesh = MeshPlot(X=X, n=n, m=m,
                 nodes=False, nodesNames=False,
                 elementsNames=False, edgesNames=False)
        
            # Update test, different precision (n,m)
        n=10 ; m=10
        X = AA.buildMesh(n=n,m=m)
        # Plot borders
        AA.borderPlot(nx=n,ny=m,
                      show=False)
        # Plot interior points and lines
        mesh = MeshPlot(X=X, n=n, m=m,
                 nodes=False, nodesNames=False,
                 elementsNames=False, edgesNames=False)

        if True:
            # Exports
            Ex = Export(X=X,n=n,m=m)
            Ex.exportFile('msh')
        



