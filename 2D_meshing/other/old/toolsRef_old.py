# Abstract
"""
S. Cadet
19/04/2022
-- Tools MeshCPU Mesh Generation
"""

# Nassim's comments [TODO]
"""
-- return at the end of each function                                   DONE
-- Noms de fichiers courts et sans espace ex: "meshSu2"                 DONE
-- fermer les fichiers quand ouvert ex: f.close() à la fin d'un open    DONE
-- Comments for properties                                              DONE
@property
def N(self): 
"" Number of Nodes""
return self.n * self.m
-- __repr__, __str__ to write                                           DONE
-- files name: toolsRef.py, meshRef_su2.su2, meshRef_vtk.vtk            DONE
-- close each file opened (ex: end of exportSU2)                        DONE
"""


# Imports
from ImportAllS import *
import ExportImport as EI



# =============================================================================
# Class ClassBasic for Class setup
# =============================================================================
class ClassBasic:
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        m = '-- Class %s\n'%(self.__class__.__name__)
        for k,v in vars(self).items():
            m+= '%s = \n%s\n'%(k,v)
        return m
    def __init__(self):
        pass



# =============================================================================
# Class MeshCPU for reference mesh
# =============================================================================
class MeshCPU(ClassBasic,EI.ImportExport):
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
    
    A = MeshCPU(n=,m=)
    or
    A = MeshCPU(n=,m=,build=False)
    """
    # Initialization
    def __init__(self,n=None,m=None,build=True):
        """
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
        # self.edgesPos = None
        # self.edgesHor = None
        # self.edgesVer = None
        
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
    
    # Export X_ref, elem or edges
    def export_X_ref(self): return self.X_ref
    def export_elem(self): return self.elem
    def export_edges(self): return self.edges
    
    # Updates the value of n or m
    def update(self,n=None,m=None,build=True):
        if n is not None: self.n = n
        if m is not None: self.m = m
        if build: self.build()
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
        return self.edges
    
    def determineEdgesPos(self):
        """ Gives the position of the start of the vertical edges from the edge array """
        for i in range(1,len(self.edges)):
            if self.edges[i,0] == 0:
                self.edgesPos = i
        self.edgesHor = self.edges[:self.edgesPos]
        self.edgesVer = self.edges[self.edgesPos:]
        return self.edgesPos 
        
    
    # Retrieve coordinates of Nodes
    def coordinatesNodes(self,k):
        """
        --- Gives the coordinates of Nodes k
        A.coordinatesNodes(k)
        or
        A.coordinatesNodes(k)[:,1:]     to only get the coordinates
        
        -- Input
        k               : (1,n)     array Node numbers, n = len(k)
            k can be:
            -manual (k1,k2,k3, ..., kn)     k1, k2, ..., kn = Node number 1, 2, ..., n
            -A.generateElements()[K]        K = Element number
            -A.generateEdges()[K]           K = Edge number
            
        -- Output
        X_ref_K         : (3,n)     array of coordinates of Nodes k array, first column is Node number
        
        [[k1  x1  y1]
         [k2  x2  y2]
              :
              :
         [kn  xn  yn]]
        
        """
        self.generateNodes()
        X_ref_K = np.zeros((len(k),3))
        for i in range(0,len(k)):
            X_ref_K[i,0] = k[i]
            X_ref_K[i,1] = self.X_ref[k[i],0]
            X_ref_K[i,2] = self.X_ref[k[i],1]
        return X_ref_K
    
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
        return k
    
    
    # Lists Nodes
    def generateListNodesNodes(self):
        """
        --- Creates the lists of every Nodes linked by an edge with the Nodes, for every Nodes
        A.generateListNodesNodes(self):
        
        -- Input
            - requires you to have generated the Nodes, Edges and Elelements with 
              A.generateNodes(), A.generateElements() and A.generateEdges() beforehand
        
        -- Output
        self.listNodesEdges         : list      a list
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
        self.listNodesEdges         : list      a list
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
        self.listNodesElements        : list      a list
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
        self.listEdgesElements      : list      a list
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
        self.listElementsEdges      : list      a list
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
        self.listElementsElements      : list      a list
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
        
        
        -- Ouput
        listResult      : array (L,l)
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
            
        elif forEach == 'Elements':
            if get == 'Nodes': listResult = self.elem
            elif get == 'Edges': listResult = self.generateListElementsEdges()
            elif get == 'Elements': listResult = self.generateListElementsElements()
        return listResult


    # Plot nodes, elements and edges
    def plot_hori(self):
        """ Draws horizontal lines between points of reference mesh """
        for p in range(0,self.m):
            x = []
            y = []
            for k in range( self.n*p , self.n*(p+1) ):
                x.append(self.X_ref[k][0])
                y.append(self.X_ref[k][1])
            plt.plot(x,y,c='black',zorder=1)
        return
    
    def plot_vert(self):
        """ Draws vertical lines between points of reference mesh """
        for p in range(0,self.n):
            x = []
            y = []
            for k in range( p , self.N, self.n):
                x.append(self.X_ref[k][0])
                y.append(self.X_ref[k][1])
            plt.plot(x,y,c='black',zorder=1)
        return
    
    def plot_nodes(self):
        """ Draws the nodes of reference mesh """
        plt.plot(self.X_ref[:,0],self.X_ref[:,1],marker='+',c='r',zorder=2,linestyle='None',markersize=15)
        return
        
    def plot_nodes_names(self):
        """ Types the numbers of the Nodes on the plot """
        for i in range(0,self.N):
            plt.annotate(i, (self.X_ref[i,0], self.X_ref[i,1]))
        return
    
    def plot_elements_names(self):
        """ Types the numbers of the elements on the plot """
        for i in range(0,self.N_el):
            coordEl = self.coordinatesNodes(self.elem[i])[:,1:]
            plt.annotate(i, ((coordEl[0,0] + coordEl[1,0])*0.98/2, (coordEl[0,1] + coordEl[3,1])*0.98/2))
        return
    
    def plot_edges_names(self):
        """ Types the numbers of the edges on the plot """
        for i in range(0,self.N_ed):
            coordEl = self.coordinatesNodes(self.edges[i])[:,1:]
            plt.annotate(i, ((coordEl[0,0] + coordEl[1,0])*1/2, (coordEl[0,1] + coordEl[1,1])*1/2)) 
        return
    
    def plot(self,size_x=5,size_y=5,nodes=True,nodesNames=True,elementsNames=True,edgesHorizontal=True,edgesVertical=True,edgesNames=True):
        """
        --- Gives a representation of the reference mesh with nodes and segments
        A.plot(size_x=,size_y=)
        
        -- Input
        size_x          : float          Physical horizontal size of plot      (default: 5)
        size_y          : float          Physical vertical size of plot        (default: 5)
        nodes           : boolean        Draw physical Nodes on plot ?         (default: True)
        nodesNames      : boolean        Show Nodes names on plot ?            (default: True)
        elementsNames   : boolean        Show elements names on plot ?         (default: True)
        edgesHorizontal : boolean        Draw horizontal edges on plot ?       (default: True)
        edgesVertical   : boolean        Draw vertical edges on plot ?         (default: True)
        edgesNames      : boolean        Show edges names on plot ?            (default: True)
        """
        # Setup
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.rcParams["figure.figsize"] = (size_x, size_y)
        self.generateNodes()
        
        # Drawing 
        if edgesHorizontal: self.plot_hori()
        if edgesVertical: self.plot_vert()
        if nodes: self.plot_nodes()
        if nodesNames: self.plot_nodes_names()
        if elementsNames: self.plot_elements_names()
        if edgesNames: self.plot_edges_names()
        
        # End
        plt.show()
        return 



# =============================================================================
# Class Lagrange for Lagrange Interpolation
# =============================================================================
class Lagrange(ClassBasic):
    """
    --- Aims to determine a general function linking each points with each others
    """
    
    #Initialization
    def __init__(self,xi,yi,build=True):
        # Setup
        X = sp.symbols('X') 
        
        # Data
        self.xi = xi
        self.yi = yi
        self.N = len(xi)
        
        # Objects
        self.function = 0
        
        # Build
        if build: self.build()
        return
        
    def build(self):
        for i in range(0,self.N):
            temp = 1
            for j in range(0,self.N):
                if i != j:
                    temp *= (X - self.xi[j]) / (self.xi[i] - self.xi[j])
            self.function += self.yi[i]*temp
        return self.function



# =============================================================================
# Main
# =============================================================================
if __name__=='__main__':
    if True:
        # Setup
        n = 5
        m = 4
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
        print("Coordinates of Nodes of an element:\n{}\n".format(ref.coordinatesNodes(ref.listAround(forEach='Elements',get='Nodes')[0])))
        
        # Edges
        print("Total amount of Edges:\n{}\n".format(ref.N_ed))
        print("List of Nodes for each Edges:\n{}\n".format(ref.listAround(forEach='Edges',get='Nodes')))
        
        # Lists
        print("List of get around each forEach:\n{}\n".format(ref.listAround(forEach='Nodes',get='Nodes')))
        
        # Plot
        ref.plot()
        ref.plot(nodes=False,nodesNames=False,elementsNames=False,edgesHorizontal=True,edgesVertical=True,edgesNames=False)
        
        # Exports/Imports
        ref.exportFile('msh')
        ref.importFile('msh')
        

        '''# Lagrange
        print(ref.listAround(forEach='Edges',get='Nodes')[0,0])
        xi = [ref.listAround(forEach='Nodes')[ref.listAround(forEach='Edges',get='Nodes')[0,0]][0],
              ref.listAround(forEach='Nodes')[ref.listAround(forEach='Edges',get='Nodes')[0,1]][0]]
        yi = [ref.listAround(forEach='Nodes')[ref.listAround(forEach='Edges',get='Nodes')[0,0]][1],
              ref.listAround(forEach='Nodes')[ref.listAround(forEach='Edges',get='Nodes')[0,1]][1]]
        print(xi,yi)
        lag = Lagrange(xi,yi)
        print(lag.build())'''



