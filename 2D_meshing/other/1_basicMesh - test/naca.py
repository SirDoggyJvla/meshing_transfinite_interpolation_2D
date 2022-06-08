# Import
from mesh_1 import *


# ---- r_bottom
# NACA Profile
X = genNACA_00t(n=50,chord=5)

# Points
# x = np.vstack([[5.1,0],[6,0],[10,0]])
ii = np.linspace(0.01,10,100)+5
x = np.vstack([[ii[k],0] for k in range(len(ii))])
X = gF.concatRow([X,x])
# gF.prePlot(equalAxis=True)
# plt.scatter(X[:,0],X[:,1],s=10); plt.show()

# Cubic Splines Interpolation
A = CubicSplines_d(M=X.copy())
MM = A.f(np.linspace(0,1,100))

# r_bottom
def r_bottom(x):
    return A.f(x)


    
# Plot
if False: 
    gF.prePlot(equalAxis=True)
    plt.title('r_bottom')
    plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
    plt.scatter(MM[:,0],MM[:,1],s=10,c='red',zorder=2)
    plt.show()

# ---- r_top
# Circular Arc
n = 50; r=10 ; d=2 
X = np.zeros((n,d)); t = np.linspace(np.pi/2,np.pi,n)[::-1]
X[:,0] = r*np.cos(t)
X[:,1] = r*np.sin(t)

# Points
ii = np.linspace(0.01,15,100)
x = np.vstack([[ii[k],10] for k in range(len(ii))])
X = gF.concatRow([X,x])

# Cubic Splines Interpolation
B = CubicSplines_d(M=X.copy())
MM = B.f(np.linspace(0,1,100))

# r_bottom
def r_top(x):
    return B.f(x)

# Intermediate curve bottom
eta =0.1; nn=100
xi  = np.linspace(0,1,nn)
d   = np.linalg.norm(r_top(xi)-r_bottom(xi),axis=1)
u = r_bottom(xi)
dd = A.normal(xi)
C = np.zeros(dd.shape)

for i in range(nn):
    C[i,:] = u[i,:]+eta*d[i]*dd[i,:]
    
# Plot
if True: 
    gF.prePlot(equalAxis=True)
    plt.title('r_bottom')
    plt.scatter(u[:,0],u[:,1],s=10,c='red',zorder=2,label='bottom')
    plt.scatter(C[:,0],C[:,1],s=20,c='green',zorder=3,label='r1')
    plt.legend()
    plt.show()
quit

# Plot
if False: 
    gF.prePlot(equalAxis=True)
    plt.title('r_top')
    plt.scatter(X[:,0],X[:,1],s=20,c='black',zorder=1)
    plt.scatter(MM[:,0],MM[:,1],s=10,c='red',zorder=2)
    plt.show()

# ---- Generate TFI Mesh
# TFI_2D and mapping
AA = TFI_2D(r_top=r_top,r_bottom=r_bottom)

# New Mesh
mCPU.setMapping(r=AA.meshPhys)
mCPU.buildMesh()
AA.plot(show=False)
mCPU.plot(init=False)

# mCPU.update(N=50,M=50)
# AA.plot(show=False)
# mCPU.plot(init=False)

