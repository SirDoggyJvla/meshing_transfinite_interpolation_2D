# Abstract
"""
N. Razaaly
19/01/21    
-- Tools Mesh Generation
Interpolation Lagrange
"""

# Import
from ImportAll import *



# -----------------------
# Class Cubic Splines for 1D Interpolation
# -----------------------
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

# -----------------------
# Class Cubic Splines for (1D)-(d-D) Interpolation
# -----------------------
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

    
    
# -----------------------
# Class Transfinite Interpolation
# -----------------------
class TFI_2D(gF.ClassBasic):
    """
    -- Return Mapping from Computational Grid (xi,eta) to Physical Grid (x,y), using 4 BCs
    A = TFI_2D(r_bottom=,r_top=,r_left=,r_right=)
    r = A.r
    x,y = r(xi=,eta=)
    """

    @property
    def A(self): return self.Corner[0,:].reshape(1,-1)
    @property
    def B(self): return self.Corner[1,:].reshape(1,-1)
    @property
    def C(self): return self.Corner[2,:].reshape(1,-1)
    @property
    def D(self): return self.Corner[3,:].reshape(1,-1)

    
    def __init__(self, r_bottom=None, r_top=None, r_left=None, r_right=None):
        """
        B ----------  D
        |             |
        |             |
        |             |
        A ----------- C

        -- Two BCs only can be defined (AB/CD) or (AC/BD). Two other ones are straight lines by default
        -- Define Smooth functions of 4 BCs defined on [0,1]: (n,) -> (n,2) array
        r_bottom               : A->C or r(xi,0)
        r_left                 : A->B or r(0,eta). r(0)=A, r(eta=1)=B
        r_top                  : B->D or r(xi,1)
        r_right                : C->D or r(1,eta). r(0)=C, r(eta=1)=D
        """
        # Basic
        self.r_x0 = r_bottom
        self.r_0e = r_left
        self.r_x1 = r_top
        self.r_1e = r_right

        # Corner A
        self.Corner = np.zeros((4,2))           # A,B,C,D
        try: self.Corner[0,:] = self.r_x0([0])
        except: self.Corner[0,:] = self.r_0e([0])

        # Corner B
        try: self.Corner[1,:] = self.r_x1([0])
        except: self.Corner[1,:] = self.r_0e([1])

        # Corner C
        try: self.Corner[2,:] = self.r_x0([1])
        except: self.Corner[2,:] = self.r_1e([0])

        # Corner D
        try: self.Corner[3,:] = self.r_x1([1])
        except: self.Corner[3,:] = self.r_1e([1])
        
        # Regularization: if only 2BCs are given
        if r_bottom is None:
            self.r_x0 = lambda x: np.dot(1-x.reshape(-1,1),self.A) + np.dot(x.reshape(-1,1),self.C)
        if r_left is None:
            self.r_0e = lambda x: np.dot(1-x.reshape(-1,1),self.A) + np.dot(x.reshape(-1,1),self.B)
        if r_top is None:
            self.r_x1 = lambda x: np.dot(1-x.reshape(-1,1),self.B) + np.dot(x.reshape(-1,1),self.D)
        if r_right is None:
            self.r_1e = lambda x: np.dot(1-x.reshape(-1,1),self.C) + np.dot(x.reshape(-1,1),self.D)
        return

    def evalVect(self,f,x):
        """Wrapper for Function float->(d,) array"""
        return f(x)
        # return np.vstack([f(x[i]) for i in range(len(x))])
    
    @property
    def dx(self):
        """Max between BD, AC along x"""
        return max(self.D[0][0]-self.B[0][0],self.C[0][0]-self.A[0][0])

    @property
    def dy(self):
        """Max between AB, CD along y"""
        return max(self.B[0][1]-self.A[0][1],self.D[0][1]-self.C[0][1])

    
    def plot(self,nx=100,ny=100,show=True,init=True):
        """ Plot Corner + BC"""
        # Initialization
        fontsize=15;e=[0.05*self.dx,0.05*self.dy]
        colorBC='grey'; lw=2
        if init: gF.prePlot(equalAxis=True)

        # Corners
        for k in ['A','B','C','D']:
            x,y=getattr(self,k)[0]
            plt.scatter(x,y,s=10,c='black',zorder=10)
            plt.annotate(k, (x+e[0],y+e[1]), fontsize=fontsize, zorder=100,weight='bold')

        # ---- BCs
        x=np.linspace(0,1,nx)
        f_b = self.evalVect(self.r_x0,x)
        f_t = self.evalVect(self.r_x1,x)
        f_l = self.evalVect(self.r_0e,x)
        f_r = self.evalVect(self.r_1e,x)
        plt.plot(f_b[:,0],f_b[:,1],label='Bottom',lw=lw)
        plt.plot(f_t[:,0],f_t[:,1],label='Top',lw=lw)
        plt.plot(f_l[:,0],f_l[:,1],label='Left',lw=lw)
        plt.plot(f_r[:,0],f_r[:,1],label='Right',lw=lw)

        # End 
        plt.legend()
        if show: plt.show()
        return 
    
    def P_xi(self,eta=None,xi=None):
        """Projector eta -- float Version"""
        return np.dot(np.diag(1-xi),self.r_0e(eta)) + np.dot(np.diag(xi),self.r_1e(eta))

    def P_eta(self,eta=None,xi=None):
        """Projector xi -- float Version"""
        return np.dot(np.diag(1-eta),self.r_x0(xi)) + np.dot(np.diag(eta),self.r_x1(xi))

    def P_xi_eta(self,eta=None,xi=None):
        """P_xi P_eta: Composite -- float Version"""
        z = np.zeros(len(eta))
        o = np.ones(len(eta))
        return np.dot(np.diag(1-xi),self.P_eta(xi=z,eta=eta))+np.dot(np.diag(xi),self.P_eta(xi=o,eta=eta))

    def P_boolean_xi_eta(self,eta=None,xi=None):
        """P_xi + P_eta: Tensor Product -- float Version"""
        return self.P_xi(eta=eta,xi=xi)+self.P_eta(eta=eta,xi=xi)-self.P_xi_eta(eta=eta,xi=xi)

    def r(self,**opt):
        """Conformal Mapping 2D"""
        return self.P_boolean_xi_eta(**opt)

    def meshPhys(self,XX):
        """
        --- From CPU nodes to Physical nodes
        X = A.meshPhys(x)

        -- Input: 
        x            : (n,2) array
        xi=x[:,0], eta=x[:,1]
        -- Output: 
        X            : (n,2) array
        x=X[:,0], y=X[:,1]
        """
        # Initialization
        # n,d=XX.shape
        # assert d==2
        xi=XX[:,0].copy()
        eta=XX[:,1].copy()
        return self.r(eta=eta,xi=xi)
        # X = np.zeros((n,d))
        # for i in range(n):
        #     X[i,0],X[i,1]=self.r(eta=eta[i],xi=xi[i])
        # return X

            
        
# -----------------------
# Functions Lagrange Interpolation
# -----------------------
class InterpolationLagrange(gF.ClassBasic):

    @property
    def n(self): return self.M.shape[0]

    @property
    def d(self): return self.M.shape[1]
    
    def __init__(self, M=None,Points=[]):
        """
        M            : (N,d) array, d=2 or 3. 
        or Points    : List of (d,) array
        M[0]=A (xi=0), M[-1]=B (xi=1); intermediate points are sorted accordingly
        """
        if len(Points) !=0:
            M = np.vstack(Points)
        self.M = M.copy()

        # CumSum for Curvilinear Abscissa
        self.s = gF.normCumDist(M)

        # Build Interpolator
        self.build()
        return

    def build(self):
        """Build Lagrange Interpolator"""
        xi = self.s.copy()
        
        L = []

        return

    def Li(self,xi,i=0):
        """
        -- Lagrange Interpolator of order i
        
        """
        # Initialization
        vect=True
        try: n = len(xi)
        except:
            n=1
            xi = np.array(xi)
            vect=False

        # Loop
        L1 = [(xi-self.s[j])/(self.s[i]-self.s[j]) for j in range(self.n) if j!=i]
        try: L1 = np.vstack(L1)
        except: L1 = np.hstack(L1)
        return L1.prod(axis=0)


    def evaluate(self,xi):

        # Initialization
        vect=True
        try: n = len(xi)
        except:
            n=1
            xi = np.array(xi)
            vect=False

        # Loop
        r = np.zeros((n,self.d))
        for i in range(self.n):
            li = self.Li(xi,i=i)
            for k in range(self.d): 
                r[:,k] = r[:,k]+li*self.M[i,k]
        return r


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
    
# -----------------------
# Main
# -----------------------
if __name__=='__main__':

    # ---- Test Lagrange
    if False: 
        # Data Generation
        m = gF.Monitor()
        n = 50; r=2 ; d=2 ; nn=300
        X = np.zeros((n,d)); t = np.linspace(0,1,n)/1.1
        X[:,0] = r*np.cos(2*np.pi*t)
        X[:,1] = r*np.sin(2*np.pi*t)

        # Interpolation Object
        m.reset()
        A = InterpolationLagrange(M=X)
        xi = np.linspace(0,1,nn)
        m('Interpolation Built',reset=True)
        M = A.evaluate(xi)
        m('Evaluation',reset=True)
        gF.prePlot(equalAxis=True)
        plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
        plt.scatter(M[:,0],M[:,1],s=10,c='red',zorder=2)
        plt.show()
    
    # ---- Test NACA
    if True: 
        # Data Generation
        m = gF.Monitor()
        n = 20; d=2 ; nn=300 ; chord=5
        # X = np.zeros((2*n,d))
        X = np.zeros((n,d))
        x = np.linspace(0,1,n)*chord
        y = profileNACA_00t(x=x, t=12, chord=chord)
        X[:n,0] = x
        X[:n,1] = y
        # X[n:,0] = x
        # X[n:,1] = -y
        gF.prePlot(equalAxis=True)
        plt.scatter(X[:,0],X[:,1],s=10,c='black',zorder=1)
        # plt.scatter(M[:,0],M[:,1],s=10,c='red',zorder=2)
        plt.show()

        # Interpolation Object
        m.reset()
        A = InterpolationLagrange(M=X)
        xi = np.linspace(0,1,nn)
        m('Interpolation Built',reset=True)
        M = A.evaluate(xi)
        m('Evaluation',reset=True)
        gF.prePlot(equalAxis=True)
        plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
        plt.scatter(M[:,0],M[:,1],s=10,c='red',zorder=2)
        plt.show()

    # ---- Test Cubic-Splines 1D
    if False: 
        # Input
        n = 10;a=1
        np.random.seed(42)
        x = np.sort(np.random.random(n)*a); x[0]=0; x[-1]=a
        # x = np.linspace(0,1,n)*a
        y = np.exp(-2*x)+2*x**2
        A = CubicSplines(x=x,y=y,build=True,BC_left=2,BC_right=2,ddy_n=4.5,dy_0=-2,dy_n=3.72)
        xx = np.linspace(0,a,1000)
        fx = A.f(xx)
        ffx = np.exp(-2*xx)+2*xx**2

        dfx = A.df(xx)
        dffx= -2*np.exp(-2*xx)+4*xx

        d2fx = A.ddf(xx)
        d2ffx= 4*np.exp(-2*xx)+4
        
        # Plot Function
        plt.title('Function')
        plt.scatter(xx,fx,s=2,color='red',label='Splines')
        plt.scatter(xx,ffx,s=2,color='blue',label='True')
        plt.scatter(x,y,s=10,color='black')
        plt.legend()
        plt.show()

        # Plot Derivative
        plt.title('Derivative')
        plt.scatter(xx,dfx,s=2,color='red',label='Splines')
        plt.scatter(xx,dffx,s=2,color='blue',label='True')
        plt.legend()
        plt.show()
        
        # Plot Derivative Second
        plt.title('Second Derivative')
        plt.scatter(xx,d2fx,s=2,color='red',label='Splines')
        plt.scatter(xx,d2ffx,s=2,color='blue',label='True')
        plt.scatter(x,A.d2y,s=10,color='black',label='y" Spline')
        plt.legend()
        plt.show()

