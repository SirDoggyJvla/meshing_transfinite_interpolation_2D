# Abstract
"""
S. Cadet
19/04/2022
-- Tools MeshCPU Mesh Generation
    -- Main
"""


# Imports
from ImportAllS import *
import ExportImport as EI
from listAround import *

# =============================================================================
# Class MeshCPU for reference mesh
# =============================================================================
class MeshCPU(gF.ClassBasic,EI.ImportExport,ListAround):
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
        self.X = None
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
    
    # Export number of nodes, elements or edges
    def export_N(self): return self.N
    def export_N_el(self): return self.N_el
    def export_N_ed(self): return self.N_ed
    
    # Export X_ref, elem or edges
    """
    -- Input
    build               : boolean   Rebuilds the array to export before export (default : false)
    
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
        if n is not None: self.n = n
        if m is not None: self.m = m
        if build: self.build()
        # Update
        self.buildMesh()
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
        --- Gives the real coordinates of Nodes k
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


    # Plot nodes, elements and edges
    def plot_hori(self):
        """ Draws horizontal lines between points of reference mesh """
        for p in range(0,self.m):
            x = []
            y = []
            for k in range( self.n*p , self.n*(p+1) ):
                x.append(self.X[k][0])
                y.append(self.X[k][1])
            plt.plot(x,y,c='black',zorder=1)
        return
    
    def plot_vert(self):
        """ Draws vertical lines between points of reference mesh """
        for p in range(0,self.n):
            x = []
            y = []
            for k in range( p , self.N, self.n):
                x.append(self.X[k][0])
                y.append(self.X[k][1])
            plt.plot(x,y,c='black',zorder=1)
        return
    
    def plot_nodes(self):
        """ Draws the nodes of reference mesh """
        plt.plot(self.X[:,0],self.X[:,1],marker='+',c='r',zorder=2,linestyle='None')
        return
        
    def plot_nodes_names(self):
        """ Types the numbers of the Nodes on the plot """
        for i in range(0,self.N):
            plt.annotate(i, (self.X[i,0], self.X[i,1]))
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
    
    def plot(self,size_x=5,size_y=5,
             plotPhys=True,show=True,
             nodes=True,nodesNames=True,
             elementsNames=True,
             edgesHorizontal=True,edgesVertical=True,edgesNames=True):
        """
        --- Gives a representation of the reference mesh with nodes and segments
        A.plot(size_x=,size_y=)
        
        -- Input
        size_x          : float          Physical horizontal size of plot      (default: 5) (better not touch)
        size_y          : float          Physical vertical size of plot        (default: 5) (better not touch)
        plotPhys        : boolean        Draw physical mesh ? Else CPU drawn   (defaut: True)
        show            : boolean        Show plot or keep active ?            (defaut: True)
        nodes           : boolean        Draw physical Nodes on plot ?         (default: True)
        nodesNames      : boolean        Show Nodes names on plot ?            (default: True)
        elementsNames   : boolean        Show elements names on plot ?         (default: True)
        edgesHorizontal : boolean        Draw horizontal edges on plot ?       (default: True)
        edgesVertical   : boolean        Draw vertical edges on plot ?         (default: True)
        edgesNames      : boolean        Show edges names on plot ?            (default: True)
        """
        # Setup
        # plt.xlim([8,11])
        # plt.ylim([2.49,2.51])
        plt.rcParams["figure.figsize"] = (size_x, size_y)
        self.generateNodes()
        
        # Physical mesh or CPU mesh ?
        if plotPhys:
            try: 
                if self.X.all() == None: 
                    self.X = self.X_ref
                    print("Physical mesh not found, ploting CPU mesh")
            except: 
                if self.X == None: 
                    self.X = self.X_ref
                    print("Physical mesh not found, ploting CPU mesh")
        else:
            self.X = self.X_ref
        
        
        # Drawing 
        if edgesHorizontal: self.plot_hori()
        if edgesVertical: self.plot_vert()
        if nodes: self.plot_nodes()
        if nodesNames: self.plot_nodes_names()
        if elementsNames: self.plot_elements_names()
        if edgesNames: self.plot_edges_names()
        
        # End
        if show: plt.show()
        return 
    
    def setMapping(self,mapTFI=None):
        """
        2D mapping s.t. 
        X = mapTFI(X_CPU)    -- from CPU mesh to physical one
        """
        self.mapTFI=mapTFI
        return

    def buildMesh(self):
        """Build Physical Mesh from Mapping and CPU Mesh"""
        try: self.X = self.mapTFI(self.X_ref)
        except: print('Mapping self.mapTFI not defined. Physical Mesh not built.')
        return 


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
        self.Corner = np.zeros((4,2))           # A,B,C,D
        # Corner A = mapTFI_bottom(xi=0,eta=0) or = mapTFI_left
        self.Corner = np.zeros((4,2))           # A,B,C,D
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
        
        
    # Calculate y for x
    def evalVect(self,f,x):
        """Wrapper for Function float->(d,) array"""
        return f(x)


    # Max distance of borders for x and y
    @property
    def dx(self):
        """Max between BD, AC along x"""
        return max(self.D[0][0]-self.B[0][0],self.C[0][0]-self.A[0][0])

    @property
    def dy(self):
        """Max between AB, CD along y"""
        return max(self.B[0][1]-self.A[0][1],self.D[0][1]-self.C[0][1])


    # Ploting
    def plot(self,nx=100,show=True,init=True):
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
        colorBC='grey'; lw=2
        if init: gF.prePlot(equalAxis=True)

        # Corners
        for k in ['A','B','C','D']:
            x,y=getattr(self,k)[0]
            plt.scatter(x,y,s=10,c='black',zorder=10)
            plt.annotate(k, (x+e[0],y+e[1]), fontsize=fontsize, zorder=100,weight='bold')

        # Borders (BCs)
        x=np.linspace(0,1,nx)
        f_b = self.evalVect(self.mapTFI_x0,x)
        f_t = self.evalVect(self.mapTFI_x1,x)
        f_l = self.evalVect(self.mapTFI_0e,x)
        f_r = self.evalVect(self.mapTFI_1e,x)
        plt.plot(f_b[:,0],f_b[:,1],label='Bottom',lw=lw)
        plt.plot(f_t[:,0],f_t[:,1],label='Top',lw=lw)
        plt.plot(f_l[:,0],f_l[:,1],label='Left',lw=lw)
        plt.plot(f_r[:,0],f_r[:,1],label='Right',lw=lw)

        # End 
        plt.legend()
        if show: plt.show()
        return 
    
    
    # Determination of each physical mesh coordinates
    """
    (x,y) = mapTFI(xi,eta)
    (x,y) = (1-xi) * mapTFI(0,eta) + xi * mapTFI(1,eta) + (1-eta) * mapTFI(xi,0)
            + eta * mapTFI(xi,1) - (1-xi) * (1-eta) * mapTFI(0,0)
            - (1-xi) * eta * mapTFI(0,1) - xi * (1-eta) * mapTFI(1,0)
            - xi * eta * r(1,1)
    Avec pour rappel:
        mapTFI_bottom               : A->C or mapTFI(xi,0)
        mapTFI_left                 : A->B or mapTFI(0,eta). mapTFI(0)=A, mapTFI(eta=1)=B
        mapTFI_top                  : B->D or mapTFI(xi,1)
        mapTFI_right                : C->D or mapTFI(1,eta). mapTFI(0)=C, mapTFI(eta=1)=D
    Et:
        self.mapTFI_x0 = mapTFI_bottom
        self.mapTFI_0e = mapTFI_left
        self.mapTFI_x1 = mapTFI_top
        self.mapTFI_1e = mapTFI_right
    Donc
        self.mapTFI_x0 = mapTFI(xi,0)
        self.mapTFI_0e = mapTFI(0,eta)
        self.mapTFI_x1 = mapTFI(xi,1)
        self.mapTFI_1e = mapTFI(1,eta)
    """
    def P_xi(self,eta=None,xi=None):
        """Projector eta -- float Version"""
        return np.dot(np.diag(1-xi),self.mapTFI_0e(eta)) + np.dot(np.diag(xi),self.mapTFI_1e(eta))

    def P_eta(self,eta=None,xi=None):
        """Projector xi -- float Version"""
        return np.dot(np.diag(1-eta),self.mapTFI_x0(xi)) + np.dot(np.diag(eta),self.mapTFI_x1(xi))

    def P_xi_eta(self,eta=None,xi=None):
        """P_xi P_eta: Composite -- float Version"""
        z = np.zeros(len(eta))
        o = np.ones(len(eta))
        return np.dot(np.diag(1-xi),self.P_eta(xi=z,eta=eta))+np.dot(np.diag(xi),self.P_eta(xi=o,eta=eta))

    def P_boolean_xi_eta(self,eta=None,xi=None):
        """P_xi + P_eta: Tensor Product -- float Version"""
        return self.P_xi(eta=eta,xi=xi)+self.P_eta(eta=eta,xi=xi)-self.P_xi_eta(eta=eta,xi=xi)

    def fonctionTest(self,eta=None,xi=None):
        """
        Calcul
        """
        # P_xi
        A = np.dot(np.diag(1-xi),self.mapTFI_0e(eta)) + np.dot(np.diag(xi),self.mapTFI_1e(eta))
        # P_eta
        B = np.dot(np.diag(1-eta),self.mapTFI_x0(xi)) + np.dot(np.diag(eta),self.mapTFI_x1(xi))
        
        z = np.zeros(len(eta))
        o = np.ones(len(eta))
        Z = np.dot(np.diag(1-eta),self.mapTFI_x0(z)) + np.dot(np.diag(eta),self.mapTFI_x1(z))
        O = np.dot(np.diag(1-eta),self.mapTFI_x0(o)) + np.dot(np.diag(eta),self.mapTFI_x1(o))
        # P_xi_eta
        C = np.dot(np.diag(1-xi),self.P_eta(xi=z,eta=eta))+np.dot(np.diag(xi),self.P_eta(xi=o,eta=eta))
        result = A + B - C
        print("ICI")
        return result


    def meshPhys(self,XX):
        """
        --- From CPU nodes to Physical nodes
        X = A.meshPhys(x)

        -- Input
        XX           : (n,2) array      the coordinate array of CPU mesh
        xi=XX[:,0], eta=XX[:,1]
        
        -- Output
        result       : (n,2) array      result of the computation
        x=result[:,0], y=result[:,1]
        """
        xi=XX[:,0].copy()
        eta=XX[:,1].copy()
        # return self.P_boolean_xi_eta(eta=eta,xi=xi)
        """
        Computation
        """
        # P_xi
        A = np.dot(np.diag(1-xi),self.mapTFI_0e(eta)) + np.dot(np.diag(xi),self.mapTFI_1e(eta))
        # P_eta
        B = np.dot(np.diag(1-eta),self.mapTFI_x0(xi)) + np.dot(np.diag(eta),self.mapTFI_x1(xi))
        
        z = np.zeros(len(eta))
        o = np.ones(len(eta))
        Z = np.dot(np.diag(1-eta),self.mapTFI_x0(z)) + np.dot(np.diag(eta),self.mapTFI_x1(z))
        O = np.dot(np.diag(1-eta),self.mapTFI_x0(o)) + np.dot(np.diag(eta),self.mapTFI_x1(o))
        # P_xi_eta
        C = np.dot(np.diag(1-xi),self.P_eta(xi=z,eta=eta))+np.dot(np.diag(xi),self.P_eta(xi=o,eta=eta))
        result = A + B - C
        print("ICI")
        return result



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

def genNACA_00t(n=10, t=12, chord=1):
    """
    X = genNACA_00t(n=,t=,chord=), X = (n,2) array
    """
    X = np.zeros((n,2))
    x = np.linspace(0,1,n)*chord
    y = profileNACA_00t(x=x, t=t, chord=chord)
    X[:,0] = x
    X[:,1] = y
    return X


# =============================================================================
# Spline interpolation on border plot
# =============================================================================
def splinePlot(X,MM,title):
    plt.title(title)
    plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
    plt.scatter(MM[:,0],MM[:,1],s=10,c='red',zorder=2)
    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__=='__main__':
    if False:
        # CPU mesh and tools
        
        # Setup
        n = 30; m = 20
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
        ref.plot(plotPhys = False,nodesNames=False,elementsNames=False,edgesNames=False)
        #ref.plot(nodes=False,nodesNames=False,elementsNames=False,edgesHorizontal=True,edgesVertical=True,edgesNames=False)

        """# Exports/Imports
        ref.exportFile('msh')
        ref.importFile(format='msh')"""
        
        
        print("DONE")
        
    if True:
        # Physical mesh example: NACA
        
        # Setup
        n = 30 ; m=20
        
        """
        We need first to define the Boundary Conditions BC (borders).
        We define bottom top left right but we could define only two.
        The important part is knowing the corners, 
        this is obtained via 2 border that don't have a common corner
        bottom = NACA profile
        top = circular arc
        left, right = random circular arc shape
        """
        # bottom border
            # NACA Profile
        X = genNACA_00t(n=50,chord=5)

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
        n = 50; r=5 ; d=2 
        X = np.zeros((n,d)); t = np.linspace(np.pi/2,np.pi,n)[::-1]
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
        n = 50; r=1 ; d=2 ; p=5/4
        X = np.zeros((n,d)); t = np.linspace(np.pi+np.pi/n,2*np.pi,n)[::-1]
        X[:,0] = (r*np.cos(t)-r)*p
        X[:,1] = (r*np.sin(t)/3)*p
        
        x = np.zeros((n,d)); t = np.linspace(0,np.pi,n)[::]
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
        n = 50; r=1 ; d=2 ; p=5/4
        X = np.zeros((n,d)); t = np.linspace(2*np.pi,np.pi+np.pi/n,n)[::]
        X[:,1] = (r*np.cos(t)-r)*p+5
        X[:,0] = (r*np.sin(t)/2)*p+10
        
        x = np.zeros((n,d)); t = np.linspace(np.pi,0,n)[::-1]
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
        Thanks to the Boundary Conditions, we can define the real
        """
        # Generate TFI Mesh
            # Setup
        ref = MeshCPU(n,m)
        
            # CPU Plot
        ref.plot(plotPhys = False,nodes=False,nodesNames=False,elementsNames=False,edgesNames=False)
        
            # TFI_2D and mapping
        AA = MeshPhys(mapTFI_top=mapTFI_top,mapTFI_bottom=mapTFI_bottom,mapTFI_left=mapTFI_left,mapTFI_right=mapTFI_right)
        
            # New Physical Mesh build and plot
        ref.setMapping(mapTFI=AA.meshPhys)
        ref.buildMesh()
        print("ok")
        AA.plot(show=False)
        ref.plot(nodes=False,nodesNames=False,elementsNames=False,edgesNames=False)
        
            # Update test, different precision
        '''n=20 ; m=10
        ref.update(n=n,m=m)
        AA.plot(show=False)
        ref.plot(nodes=False,nodesNames=False,elementsNames=False,edgesNames=False)
        '''
        
        print("DONE")
        