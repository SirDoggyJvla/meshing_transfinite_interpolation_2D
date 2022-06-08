# Abstract
"""
N. Razaaly
19/01/21    
First Mesh Generation
"""


# Import
from tools_mesh import *

# -----------------------
# Class
# -----------------------
class Mesh2D(gF.ClassBasic):

    @property
    def n(self):
        return self.X.shape[0]

    def initMeshCPU(self):
        N = self.N; M = self.M
        h = 1.0/(N-1); kk = 1.0/(M-1)
        X_CPU = np.zeros((N*M,2))
        for i in range(N):
            for j in range(M):
                X_CPU[i+j*N,:] = [i*h,j*kk]
        return X_CPU
    
    def __init__(self, edges=None, elements=None, N=None, M=None, X_CPU=None, X=None):
        """
        X_CPU             : (N,2) array - Mesh in Unit Square
        N, M              : Integers such as
Nodes[i+j*N,:] = X_i_j  -> If X_CPU not given, generic nodes are generated: 
X_CPU[i+j*N,:] = [i*h,j*kk], h=1/(N-1),k=1/(M-1)
        edges, elements   : Optional
Line 0: (X_i0 X_i+1_0)...
        """
        # Structured
        self.N = N
        self.M = M

        # Computational Mesh
        try:    self.X_CPU  = X_CPU.copy()
        except: self.X_CPU = self.initMeshCPU()

        # Physical Mesh
        self.X=None
        try: self.X = X.copy()
        except: pass
        
        # All
        self.Ed = edges
        self.El = elements

        # Edges
        self.buildEdges()
        return

    def num(self,i,j):
        """Numerotation (i,j)->k=i+N*j"""
        return i+self.N*j

    def numInv(self,k):
        """Numerotation k=i+N*j -> (i,j)"""
        return k%self.N,int(k/self.N)
    
    def buildEdges(self,force=False):
        """
        -- Build Edges for Structured Grid is None
        From Line 0, to Line M-1
        Then Columns 0, to Columns N-1
        
        Matrix of Integers
        """
        if self.Ed is not None and force==False: return

        # Lines
        Line = []
        for j in range(self.M):
            for i in range(self.N-1):
                Line.append([self.num(i,j),self.num(i+1,j)])
        Line = np.vstack(Line)
        nn = Line.shape[0]

        # Columns
        Col = []
        for i in range(self.N):
            for j in range(self.M-1):
                Col.append([self.num(i,j),self.num(i,j+1)])
        Col = np.vstack(Col)
        mm = Col.shape[0]

        # Concat
        self.Ed = np.zeros((nn+mm,2),dtype=int)
        self.Ed[:nn,:] = Line
        self.Ed[nn:,:] = Col
        return 

    @property
    def nEdge(self):
        try:
            return self.Ed.shape[0]
        except:
            return 0

    def plotCPU(self,**opt):
        """ Plot CPU Mesh-- Options in self.plot"""
        self.plot(X=self.X_CPU)
        return 

    def update(self,N=None,M=None):
        """
        Automatic Regeneration of X_CPU
        if mapping defined, X updated by self.r(X_CPU)
        """
        # Initialization
        if N is None and M is None: return 
        if N is not None: self.N=N
        if M is not None: self.M=M

        # Update
        self.X_CPU = self.initMeshCPU()
        self.buildEdges(force=True)
        self.buildMesh()
        return 
        
    def setMapping(self,r=None):
        """
        2D mapping s.t. 
        X = r(X_CPU)    -- from CPU mesh to physical one
        """
        self.r=r
        return

    def buildMesh(self):
        """Build Physical Mesh from Mapping and CPU Mesh"""
        try: self.X = self.r(self.X_CPU)
        except: print('Mapping self.r not defined. Physical Mesh not built.')
        return 
        
    def plot(self,s=5,color='black',lw=1,colorLine='blue',init=True,X=None,show=True):

        if X is None: X=self.X
        
        # Init
        if init: gF.prePlot(equalAxis=True)

        # Nodes
        plt.scatter(X[:,0],X[:,1],s=s,c=color,zorder=1)

        # Edges
        for k in range(self.nEdge):
            A = X[self.Ed[k,0],:]
            B = X[self.Ed[k,1],:]
            plt.plot([A[0],B[0]],[A[1],B[1]],lw=lw,color=colorLine,zorder=0)
        
        if show: plt.show()
        return


# -----------------------
# Create Computational Domain Mesh
# -----------------------
if True: 
    # Datas
    N = 30; M=20

    # Mesh
    mCPU = Mesh2D(N=N,M=M)
    mCPU.plotCPU()
    X_CPU=mCPU.X_CPU.copy()
    
# -----------------------
# TFI 2D: NACA Profile + BC
# -----------------------
if __name__=='__main__': 
    # ---- r_bottom
    # NACA Profile
    X = genNACA_00t(n=50,chord=5)

    # Points
    # x = np.vstack([[5.1,0],[6,0],[10,0]])
    ii = np.linspace(0.01,5,100)+5
    x = np.vstack([[ii[k],0] for k in range(len(ii))])
    X = gF.concatRow([X,x])
    # gF.prePlot(equalAxis=True)
    # plt.scatter(X[:,0],X[:,1],s=10); plt.show()

    # Cubic Splines Interpolation
    A = CubicSplines_d(M=X.copy())
    MM = A.f(np.linspace(0,1,100))    # <------------ CPU ?

    # r_bottom
    def r_bottom(x):
        return A.f(x)

    # Plot
    gF.prePlot(equalAxis=True)
    plt.title('r_bottom')
    plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
    plt.scatter(MM[:,0],MM[:,1],s=10,c='red',zorder=2)
    plt.show()

    # ---- r_top
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
    MM = B.f(np.linspace(0,1,100))    # <------------ CPU ?

    # r_top
    def r_top(x):
        return B.f(x)

    # Plot
    gF.prePlot(equalAxis=True)
    plt.title('r_top')
    plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
    plt.scatter(MM[:,0],MM[:,1],s=10,c='red',zorder=2)
    plt.show()
    
    # ---- r_left
    # Circular Arc
    n = 50; r=1 ; d=2 ; p=5/4
    X = np.zeros((n,d)); t = np.linspace(np.pi+np.pi/n,2*np.pi,n)[::-1]
    X[:,0] = (r*np.cos(t)-r)*p
    X[:,1] = (r*np.sin(t)/3)*p
    
    x = np.zeros((n,d)); t = np.linspace(0,np.pi,n)[::]
    x[:,0] = ((r*np.cos(t)-r*2)-r)*p
    x[:,1] = (r*np.sin(t)/3)*p

    # Points
    # ii = np.linspace(0.01,10,100)
    # x = np.vstack([[ii[k],5] for k in range(len(ii))])
    X = gF.concatRow([X,x])

    # Cubic Splines Interpolation
    C = CubicSplines_d(M=X.copy())
    MM = C.f(np.linspace(0,1,100))    # <------------ CPU ?

    # r_left
    def r_left(x):
        return C.f(x)

    # Plot
    gF.prePlot(equalAxis=True)
    plt.title('r_left')
    plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
    plt.scatter(MM[:,0],MM[:,1],s=10,c='red',zorder=2)
    plt.show()
    
    # ---- r_right
    # Circular Arc
    n = 50; r=1 ; d=2 ; p=5/4
    X = np.zeros((n,d)); t = np.linspace(2*np.pi,np.pi+np.pi/n,n)[::]
    X[:,1] = (r*np.cos(t)-r)*p+5
    X[:,0] = (r*np.sin(t)/2)*p+10
    
    x = np.zeros((n,d)); t = np.linspace(np.pi,0,n)[::-1]
    x[:,1] = ((r*np.cos(t)-r*2)-r)*p+5
    x[:,0] = (r*np.sin(t)/2)*p+10
    
    # Points
    # ii = np.linspace(0.01,10,100)
    # x = np.vstack([[ii[k],5] for k in range(len(ii))])
    X = gF.concatRow([X,x])
    x1 = np.zeros((len(X[:,0]),2))
    for i in range(0,len(X[:,0])):
        x1[i,0] = X[len(X[:,0])-i-1,0]
        x1[i,1] = X[len(X[:,0])-i-1,1]

    # Cubic Splines Interpolation
    D = CubicSplines_d(M=x1.copy())
    MM = D.f(np.linspace(0,1,100))    # <------------ CPU ?

    # r_right
    def r_right(x):
        return D.f(x)

    # Plot
    gF.prePlot(equalAxis=True)
    plt.title('r_right')
    plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
    plt.scatter(MM[:,0],MM[:,1],s=10,c='red',zorder=2)
    plt.show()
    
    
    # ---- Generate TFI Mesh
    # TFI_2D and mapping
    AA = TFI_2D(r_top=r_top,r_bottom=r_bottom,r_left=r_left,r_right=r_right)

    # New Mesh
    mCPU.setMapping(r=AA.meshPhys)
    mCPU.buildMesh()
    # XX = AA.meshPhys(X_CPU)
    # mPHY = Mesh2D(X=XX,N=N,M=M)
    AA.plot(show=False)
    mCPU.plot(init=False)
    mCPU.update(N=20,M=10)
    AA.plot(show=False)
    mCPU.plot(init=False)

