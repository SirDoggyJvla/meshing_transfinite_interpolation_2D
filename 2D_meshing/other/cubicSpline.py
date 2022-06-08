# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:33:46 2022

@author: utilisateur
"""
from ImportAllS import *


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