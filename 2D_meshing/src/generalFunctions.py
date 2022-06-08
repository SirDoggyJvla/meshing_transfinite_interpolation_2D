# -*- coding: utf-8 -*-
"""
Date:      30/03/2017
Update:    30/03/2017
Author:    Nassim RAZAALY 
Institute: INRIA Bordeaux Sud-Ouest
Project:   
File:      General Functions used everywhere
           -> LHS
           -> Dictionary Initialization
           -> 2D plot functions
Python:    Python ver. 3.5
Note:      High Quality Plots: https://www.bastibl.net/publication-quality-plots/
           Customize Plot Matplotlib: http://matplotlib.org/users/customizing.html
import numpy as np
import matplotlib as mpl
mpl.use('pdf')

# HiDPI
matplotlib.rcParams['figure.dpi'] = 200  

# Font size
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Customize format ticks
from matplotlib.ticker import FuncFormatter
def format_tick_labels(x, pos): return '{0:.2f}'.format(x)   
ax1.xaxis.set_major_formatter(FuncFormatter(format_tick_labels))    

# Plot Matrix Values
M = np.random.random((100,100))
fig, ax = plt.subplots()
cax = ax.matshow(np.real(M))
cbar = fig.colorbar(cax)
ax.set_title('Real part of Fourier matrix')
fig.show() 


# with was measured in inkscape
width = 3.487
height = width / 1.618

fig, ax = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

x = np.arange(0.0, 3*np.pi , 0.1)
plt.plot(x, np.sin(x))

xmin,xmax = plt.xlim()  # Get range bounds

ax.set_ylabel('Some Metric (in unit)')
ax.set_xlabel('Something (in unit)')
ax.set_xlim(0, 3*np.pi)

fig.set_size_inches(width, height)
fig.savefig('plot.pdf')

Print String with space: m = '%s'%('Nassim'.ljust(50))
print('Test=',end='',flush=True)        # No \n, show Directly to output

Class Techniques: 
    # Dynamic Properties [folders]
    @property
    def folderPrivate(self):
        return '%s%s/private/'%(self.folderDat,self.opt['blade'])

ClassTemplate Ready to use: Requires self.folder, self.listData

# Lambda Functions
f = [lambda x: i*x for i in range(10)] 
f = lambda x: 5*x**2


# Easy Import
FOLDER_LIB = '/home/energia3/isammarco/Documents/NASSIM/CFX/libCFX/'
sys.path.insert(1,FOLDER_LIB)

# Quand vous en avez marre des warnings
import warnings
warnings.filterwarnings("ignore")

# Training/Test sets
from sklearn.model_selection import train_test_split
xv, xt, fv, ft = train_test_split(XV, FV, test_size=0.5, random_state=42) 

# DE Optimization
opt={'popsize': 20,'maxiter':100,'polish': True, 'seed': 42}#, 'tol': 1e-3, 'atol': 1e-5}
import scipy
def f(x):
    if len(x.shape)==1: x = x.reshape(1,-1)
    return np.sum(x,axis=1)
d=11 ; NMAX = 3000 ; xmin = -2*np.ones(d) ; xmax = 2*np.ones(d)
nmax = opt['popsize']*(opt['maxiter']+1)*d
# or
maxiter = NMAX/d/opt['popsize']-1
opt['maxiter'] = int(maxiter)
bounds  = []
for k in range(d):
    bounds.append([xmin[k],xmax[k]])
res     = scipy.optimize.differential_evolution(f, bounds, **opt)
print(res)
res['x'], res['fun'], res['nfev']                                              

# COBYLA
# Constraints: g_i(x)>0
def objective(x): return x[0]*x[1]
def constr1(x):   return 1 - (x[0]**2 + x[1]**2)
def constr2(x):   return x[1]
res = ss.optimize.fmin_cobyla(objective, [0.0, 0.1], [constr1, constr2], rhoend=1e-7,disp=True)
print(res)
                                                                               # Non-Local Variable
def outer():
    x = "local"
    def inner():
        nonlocal x
        x = "nonlocal"

# GrandParent inheritance
class GrandParent(object): 
    def m(self):
        print('Grandparent')

class parent(GrandParent):
    def m(self):
        print('parent')

class child(parent):
    def m(self):
        print('Hi GrandPa')
        super(parent,self).m() 

A = child()
A.m()
"""
# from matplotlib import rc,rcParams
# # activate latex text rendering
# rc('text', usetex=True)
# rc('font', weight='bold')
# rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

# # Pre-Plot
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
# plt.rc('font', weight='bold')
# plt.rc('legend', fontsize=tickSize)
# plt.rc('axes', labelsize=tickSize)
# fig,ax = plt.subplots()
# # ax.tick_params(axis = 'both', which = 'major', labelsize = 24)               # ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)
# ax.tick_params(axis = 'both', labelsize = tickSize)

# k parmi n: from math import comb ; p = comb(10,3)

# ------------------------------------------------------
# Import 
# ------------------------------------------------------
# Import Main Modules in path

# Basics Packages
#from __future__ import print_function  # So print(m, end = '') is callable
import sys, os, time, inspect
from pyDOE.doe_lhs import lhs
import numpy as np
import time, datetime
import code   #code.interact(local=dict(globals(), **locals()))
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
from inspect import getframeinfo, stack, currentframe
from itertools import cycle
from scipy.spatial import Delaunay, ConvexHull, distance
import scipy.stats as ss
from scipy import sparse
from six.moves import input
from sklearn.cluster import KMeans
from copy import deepcopy
try: import minepy
except: print('generalFunctions.py: Failed to load minepy.\nuse "pip install minepy"')
from math import sin, cos, tan, atan, sqrt

# HiDPI Matplotlib
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200


# ------------------------------------------------------
# Parameters - Datas
# ------------------------------------------------------
# Color List= ["red", "blue", "green", "orange", "black", "grey", "yellow"] then all others
from matplotlib import colors as mcolors
colorList   = ["red", "blue", "green", "orange", "black", "grey", 'purple']
# colorList   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# colorList  += dict(mcolors.CSS4_COLORS).keys()
colorCycle  = cycle(colorList)
# next(colorCycle)

def colorSet(n,no=[]):
    """
    Set of Colors, excluding colors in no
    -> s = colorSet(8,no=['black'])
    """
    col = [k for k in colorList if k not in no]
    return col[:n]

# ------------------------------------------------------
# General Functions always used
# ------------------------------------------------------
"""
Path:
  -> nameFolder(folder)
  -> importPathFolder(folder)
  -> getSubDirectory(fileName=None, folder=None, n = 1)
  -> createFolder(fold,**opt): (just wrapper!!)
  -> folderInitialize_create(fold)                      
  -> getFolder(fileName, n = 1)                          : 
  -> folderAddress(folder)
  -> listSubDir(fileName=None, folder=None, onlyFile=False, onlyDir=True):
  -> saveNumpy(fName,x,prefix='',separator='', around='"',tab=None,tabSeparator='=', header=None):
  -> convertDataToNumpy(data, fileName = None):
  -> convertNumpyToData(A, fileName = None):
  -> extractListIndices(l,ind):
  -> loadNumpy(fName):
  -> loadNumpyLines(fName,nLines, fileTemp=None):
  -> loadNumpyBlock()
  -> convert_bytes(num):
  -> file_size(file_path):
  -> _getSizeWriteRead(o):
  -> getSize(obj)
  -> writeString(fName,message):
  -> readString(fName)
  -> initDictFull(d,dOld):
  -> extractDict(opt,name,default):
  -> funInit(name, d):
  -> lineString(name, data,find=False):
  -> modifWordFile(fileName, oldWord, newWord):

Sampling:
  -> mic_evaluate(xv,fv,method='default', **opt):
  -> sampleLHS(N, xmin, xmax)                            : 
  -> sampleUniform(N,xmin,xmax)
  -> sampleLHS_spaceAS(N0, xmin=None, xmax=None, zmin=None, zmax=None, nMC=1e5, V=None, kLHS=10, criterion=None):
  -> which_inConvexHull(p, cloud)                        : 
  -> weightedKmeans(x=None,K=None, TOL = 1e-4, NMAX = 100, f = None):
  -> kMeansCenters(u, K) : 
  -> findArrayIndice(xOld,x):
  -> chooseKpoints(x=None, K=None, xv=None):
  -> filterPoints(x,TOL=None):
  -> randomDiscreteWeight(Kn,iSel,weight)
  -> addRandom_one(xv=None, x=None, xmin=0, xmax=1, N=1e4):
  -> addRandom(xv=None,K=10,**opt):
  -> randomKfold(n,K):

Cumulative Distance:
  -> cumDist(M)                                          : t = cumDist(M)
  -> normCumDist(M)                                      : t = normCumDist(M)

Options:
  -> dictBuildDefault(dictDefault, **opt)                : 
  -> dictRemoveDefault(dictDefault, **opt)               : 
  -> dictBuildDefault2(dictDefault, **opt)               : 
  -> dictBuildDepth(dictDefault,dNew)                    : 
  -> inverseDictionary(dictIn)                           : newDict = inverseDictEntry(dictIn)
  -> dictList(d,l)                                       : 
  -> dictListValue(d,l)                                  : 
  -> extendListString(l)                                 : 
  -> cleanDictKey(name,opt)
  -> inverseArray(a)                                     : 
  -> randomInteger(ni=None,ntot=None,seed=None, xv=None, fv=None)
  -> selectColor(lColor=[],i=0)
  -> concat1D(l)                                         : 
  -> concatColumns(l):
  -> repeat1D(A,n)
  -> concatRow(l)                                        : xv = concatRow((x,y,z))
  -> findArray(a,b)                                      : c = findArray(a,b)
  -> cdist(x,y, enforce = False)                         :  
  -> columnIndexing(name,value)                          :
  -> dictToNumpy(d)
  -> partitionInteger(n,NMAX):
  -> delArray1D(ori,ind):
  -> delArray2D(ori,ind)

Maths: 
  -> interp1D(x,y)
  -> interp(x,y)
  -> solveSecondDegree(a,b,c):
  -> integrate(x,f):
  -> boundaryLayerParameters(u=None,y=None, fileInput=None, delimiter=','):
  -> gramSchimdt(base)
  -> rankGS(V,v):
  -> rankGS_list(V,W):
  -> addGS(V,v,eps=1e-4):
  -> performGS(V,W,eps=1e-4):
  -> rotBlade(bladePoints, theta, center):
  -> plotTangentPerso
  -> ellipseTE(A=None, B=None, m1=None, m2=None,
              nTE=20, includeAB=False, vBig=1e8, vSmall=1e-8, allEllipse=False, show=False, clockwise=False):
  -> ellipse_xy(C=[0,0], a=2, b=1, theta=0, t=None, Npts=1000):
  -> findParam_ellipse(X, Y, C=[0,0], a=2, b=1, theta=0):
  -> cubicInterp1D(x=[],y=[],dx=[],dy=[], show=False):
  -> computeB_spline1D(x, f, alpha, beta, periodic):
  -> compute_f_spline1D(splinePar, xeval):
  -> compute_df_spline1D(splinePar, xeval):
  -> computeMatrixA_spline1D(x,f):
  -> computeSnd_b_spline1D(x, f):
  -> interpSplines1D(xx,yy,alpha=0,beta=0,periodic=False):
  -> fmin_cobyla(f=None, x0=None, xmin=None, xmax=None, gCons=[], tol=1e-6, nmax=1000,disp=True, show=True):
  -> fmin_cobyla_one(f=None, x0=None, xmin=None, xmax=None, gCons=[], tol=1e-6, nmax=1000,disp=False, show=False, returnAll=False):

BFGS:
  -> interpLineSearch(df0,f0,fi,fi1,ai,ai1):
  -> backTracking_lineSearch(fun, xk, fk, dfk, pk, fk1=None,c1=1e-4,rho=0.8,modeDebug=False,aTab=[],fTab=[],nloop=50):
  -> lineSearch(fun, xk, fk, dfk, pk, fk1=None,c1=1e-4,nmax=50,eps1=1e-3,eps0=1e-3, modeDebug=False,dfk1=None,pk1=None,alpha1=None):
  -> class BFGS:

Plot:
[Global Variables]:   boolScript, nameScript, folderScript, indScript
[Global Functions]:
  -> setBoolScript(boolVal)
  -> setNameScript(name)
  -> setFolderScript(folder)
  -> setIndScript(ind)
[Global Get & All]: 
  -> getBoolScript()
  -> getNameScript()
  -> getFolderScript()
  -> getIndScript()
  -> getFileScript()
  -> getImageOutScript()
  -> createFileScript()
  -> getDataFileScript()
  -> writeFileScript(data)
  -> savePointsDataScript(points)
[Functions Plot]:
  -> initPlot(...)
  -> prePlot(left=0.15,bottom=0.16,top=0.9,right=0.99,scientific_x=False,scientific_y=True,grid=False, bold=True, powerlimit=(-3,2), tickSize=10, equalAxis=False, frameon=True, xTicks=True, yTicks=True):
  -> plotBlade(xPS=None, yPS=None, xSS=None, ySS=None, show=True, color='black', label='', save=False, fileSave='blade.eps', dpi=300, pitch=0.045, frameon=False, xTicks=False, yTicks=False, degree=-90, kPitch=[1]):
  -> addTicks(axes,value,label,pos='x', fun=None):
  -> contour2D_Plot(fun, xmin, xmax, Npts=80, cmap='jet', N_level=20):
  -> contour3D_Plot(fun, xmin, xmax, Npts=50, cmap='jet', N_level=10):
  -> scatterPlot(points_0, label = "", c = "black", s = 1, m = '.', annotate = False, e = [0,0], fontsize = 10, onlyAnnotate=False,zorder=None):
  -> plotPlot(points_0, label="", c=None, s=1, m='.'):
  -> endPlot()                                           : 
  -> plotHisto(x, N_bins=10, alpha_bins=0.3, N_x=1000, show=True, kde_plot=True):
  -> colorScatterPlot(x,z,cmap = 'jet', N_level = 20):
  -> ePlot(fileSave, save = False, show = True, extension='eps', dpi=1200, fontsize=15, markerSize=None, loc='best', legend=True):
  -> plotArraySimple(x,save=False, show=True, fileName='multi.png',dpi=300,s=1,color='black',**opt):
  -> def plotArray(x=None, color=None, marker=None, size=None, label=None,xlabel=None,
              save=False, show=True, fileName='multi.png',dpi=300, normalize=False,
              left=.1, bottom=.1, right=.99, top=.99,
              alpha=0.3,i_PDF=0,**opt):
  -> plotLSS(G=None, mpfp=None, level=None, axes=None, 
            nCol=None, nRaw=None, couples=None,colorSafe='lightgreen', colorFail='lightsalmon',
            fontsize=15,figsize=15, a=-5, b=5, Npts=100, show=True, save=False, fileSave='fig.png', dpi=300, xTicks=None, yTicks=None,
            optG = {'colors': 'black', 'linewidths': 5, 'linestyles': '-', 'zorder': 1},
            listG=[], listOpt=[], listLevel=[], optDefault={'colors': 'black', 'linewidths': 5, 'linestyles': '--', 'zorder': 1} ):
Error:
  -> getVarName(**kwargs):
  -> line_code()                                         : l = line_code()
  -> fileLine_code()                                     : (f,n) = fileLine_code()
  -> fileLine_message(**opt)                             : 
  -> message_ioError(filePy,line,readOption,fileName, **opt): 
  
Print:
  -> printDict(d)                                       
  -> printList(l)                                        : message = printList(l)
  -> printTab(x, init='', end='', initLine='x_', initVect=' = [', endVect=']', separator=', ', showLine=True):
  -> printTab1D(x, init='[', end=']', separator=', '):
  -> indiceRenumerotation(N=0, offset=0, inverse=[], remove=[], order=['remove','offset','inverse']):
  -> convertTime(t0):
  -> convertTime(t)                                      : message = convertTime(t)
message = '1j 5h 06m 10s'
  -> convertInch(f,i)                                    : meter = convertInch(foot,inch)

Numerics:
  -> gradNumFunction(f, h = 1e-4, d = None, p = None)    : gf = gradNumFunction(f, h = 1e-4, d = 3, p = 5)    f: R^d -> R^p

Error:
def mySystem(command,message=None):
def checkError(b_bool,message=''):
def checkErrorFile(fileName,message='',stop=True):
def eliminateComment(fName):
def initScript():
def endScript():
def runScript(command=None,fileScript=None,prepareOnly=False,parallel=False):


Batch:
  -> createBatch(folder=None, jobName='job', ncpu=1, nhours=1, nminutes=0, exeFile='run.sh', NCPUMAX=100, NHOURSMAX=1000, codeExe=None, createFolder=False):
  -> globalBatchScript(listBatch=[], folder=None):
  -> automaticBatch(folder=None, jobName='job_', ncpu=1, nhours=1, nminutes=0, exeFile='run.sh', NCPUMAX=100, NHOURSMAX=1000, codeExe=None, runScript=False):

Class:
class ClassBasic
class ClassTemplateMin:
class ClassTemplate:
class Monitor:
class ResultDict:

"""
def logToNormal(mu=0,sigma=1):
    m = mu ; v=sigma**2 ; 
    mu = np.log(m/(np.sqrt(1+v/m**2)))
    sigma= np.sqrt(np.log(1+v/m**2))
    return mu,sigma

def normalToLog(mu=0,sigma=1):
    m = np.exp(mu+0.5*sigma**2)
    s = np.exp(2*mu+2*sigma**2) - m**2
    return m,np.sqrt(s)

# ------------------------------------------------------
# Small Function for defining path
# ------------------------------------------------------
# Folder Name
def nameFolder(folder):
    if folder[-1] == '/': return folder
    return folder + '/'

# Function: Import Path Folder -> returns folderName
def importPathFolder(folder):
    currentFolder    = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), folder))
    if not path in sys.path:
        sys.path.insert(1, path)
    return folder
# end Function importPathFolder 

# Get Subdirectory name
def getSubDirectory(fileName=None, folder=None, n = 1):
    """
    Get Subdirectory name

    import datetime
    print datetime.datetime.now()
    
    fold = getSubDirectory(fileName, n = 1)  [Contains already '/' at the end] 
    fold = $dir(fileName)/(../../)$(n times)
    n times .. w.r.t. directory containing fileName, or folder

    Date: import datetime, datetime.datetime.now()
    Call: fold = getSubDirectory(filename='file.py', n=4)

    if 'blabla' in open('example.txt').read(): print(True)

    np.load(filef, encoding='latin1')
    preFolder = os.listdir(folderSU2)[0]
    cwd = os.getcwd() - pwd
    0=sys.exit() ; 1=sys.exit('Error') [stderr]
    folder = os.path.dirname(os.path.realpath(fileName))
    preLink = os.path.dirname(os.path.realpath(__file__))
    os.chdir(newFolder)
    re.split(r'(\d+)', preFolder) -> Separate Numbers/letters
    os.path.exists(fileN)
    os.path.isfile(fileN)
    os.path.isfolder(fileN)
    os.path.getsize(fullpathhere) > 0
    """
    # Initialization
    if folder is None:
        assert fileName is not None, fileName
        folder = os.path.dirname(os.path.realpath(fileName))
    folder = os.path.realpath(folder)

    # Loop
    pre = folder
    for i in range(n):
        pre = os.path.split(pre)[0]
    pre = pre + '/'
    return pre
# End Function getSubDirectory
# --------


# --------
# List of Files/Directories of a given directory [if fileName given, its directory is considered]
# --------
def listSubDir(fileName=None, folder=None, onlyFile=False, onlyDir=True):
    """                                                                                        
    Get List of Files/Directories of a given directory [if fileName given, its directory is considered]
    -> Call: [all directories contain already '/']
    listDir=listSubDir(folder='./folder',onlyDir=True)
    listFile=listSubDir(folder='./folder',onlyFile=True)
    l=listSubDir(folder='./folder')
    """
    # Initialization
    if folder is None:
        assert fileName is not None, fileName
        folder = os.path.dirname(os.path.realpath(fileName))
    folder = os.path.realpath(folder)
    if folder[-1] != '/': folder = folder+'/'
    listdata = os.listdir(folder)
    n=len(listdata)

    # Loop
    nameFile=[]
    nameDir=[]
    for i in range(n):
        fi = folder+listdata[i]
        if os.path.isdir(fi):
            fi=fi+'/'
            nameDir.append(fi)
        if os.path.isfile(fi): nameFile.append(fi)
    if onlyFile: return nameFile
    if onlyDir: return nameDir
    return sorted(nameFile+nameDir)
# End Function listSubDir
# --------


def createFolder(fold,**opt):
    return folderInitialize_create(fold,**opt)

# --------
# Folder "mkdir"
# --------
# Call fold = folderInitialize_create(fold)
def folderInitialize_create(fold,new=False):
    """
    Input:
    -> fold               : string - folder name to create
    -> new                : Bool - If True, New Folder anyway
    Output:
    -> fold               : idem
    """
    if not os.path.exists(fold):
        try: os.makedirs(fold)
        except: print('Failed to Create Folder [folderInitialize_create]:\n %s'%(fold))
    if new: os.system('rm -r %s && mkdir %s'%(fold,fold))
    fold = os.path.abspath(fold) + '/'
    return fold
# End function folderInitialize_create
# --------


def folderAddress(folder):
    return os.path.abspath(fold) + '/'

# -------------
# Get Folder Name from file - Given Depth
# -------------
def getFolder(fileName, n = 1):
    """                       
    getFolder(fileName, n = 1)
    Get Folder Name from file - Given Depth
    fold = getFolder(fName, n=2)  [Contains already '/' at the end]
    fold/Dat/private/fileName.dat
    Get Folder, from file within n directories (ex: flowDat, with n=2) 
    """
    pre = os.path.dirname(os.path.realpath(fileName))
    for i in range(n):
        pre = os.path.split(pre)[0]
    pre = pre + '/'
    return pre
# End Function getFolder
# --------


# -------------
# Save Numpy file - with nice comment on the data
# -------------
def saveNumpy(fName,x,prefix='',separator=None, around='',tab=None,tabSeparator='=', header=None):
    """
    Save Numpy file - with nice comment on the data

    --> Call:
    m = saveNumpy('./save.txt',x,prefix='x_',separator=',', around='"',tab=['Pin',...,'Tin'],tabSeparator='=', header=None)
    ----> First Line: m = #"x_0=Pin',...,"x_8=Tin"           file saved in save.txt
    ----> If x None, empty file created
    """
    if x is None: 
        os.system('touch %s'%(fName))
        return ''

    if len(x.shape) == 1: 
        n = len(x)
    else:   
        n=x.shape[1]

    if header is None:
        if separator is None: 
            header = ''
        else:
            m=''
            if tab is None:
                l = [around+prefix+str(i)+around for i in range(n)]
            else:   
                l = [around+prefix+str(i)+tabSeparator+str(tab[i])+around for i in range(n)]
            header=m+separator.join(l)

    np.savetxt(fName,x,header=header)
    return header
# End Function saveNumpy
# --------

# Convertion data to Numpy
def convertDataToNumpy(data, fileName = None, delimiter=None):
    """
    Convert list of Strings containing Numbers into Numpy array [using fileWriting]
    Input: 
    -> data            : List of Strings containing Numbers
    Output:
    -> A               : (N,d) array - Informations contained 
    """
    if fileName is None: fileName = './fileNumpy_%s.txt'%(np.random.randint(1000000))
    fileTemp         = fileName[:-4] + "_numpy.txt"
    writeString(fileTemp,data)
    try: A                = np.loadtxt(fileTemp,delimiter=delimiter)
    except:
        print("Error Converting Data to array. Return None")
        A = None
    os.system('rm %s'%(fileTemp))
    if len(A.shape)==1: A = A.reshape(1,-1)
    return A

# Convertion Numpy To Data
def convertNumpyToData(A, fileName = None):
    """
    Convert Numpy array to list of Strings
    Input: 
    -> A               : (N,d) array
    Output:
    -> data            : list of Strings
    """
    if fileName is None: fileName = './fileNumpy_%s.txt'%(np.random.randint(1000000))
    fileTemp         = fileName[:-4] + "_numpy.txt"
    np.savetxt(fileTemp,A)
    A = readString(fileTemp)
    os.system('rm %s'%(fileTemp))
    return A
    


def extractListIndices(l,ind):
    """
    Extraction of List from indices
    Ex: 
    ll = extractListIndices(['f0','f1','f2','f3','f4','f5'],[3,2,7])
    ll = [f3,f2]
    INput: 
    -> l               : List
    -> ind             : List of integers
    Output:
    ll containing l[ind] if ind[k] is in l
    """
    return [l[ind[k]] for k in range(len(ind)) if ind[k] < len(l)]


# -------------
# Load Numpy file 
# -------------
def loadNumpy(fName):
    """
    Load Numpy file [if empty/does not exist -> Return None]

    --> Call: x=loadNumpy(fName)
    """
    if os.path.exists(fName) is False: 
        return None
    if os.path.getsize(fName) > 0: 
        return np.loadtxt(fName)
    return None
# End Function loadNumpy
# --------

def loadNumpyLines(fName,nLines, nEnd=0, nKeep=None, fileTemp=None, delimiter=None):
    """
    -- Read File as numpy array, without accounting for the nLines first lines
    temporary file saved as such, removing the first nLines
    it is read as np.loadtxt, then erased
    -- Tutorial
    A  = loadNumpyLines('./file.dat', 3)
    A  = loadNumpyLines('./file.dat', 3, nEnd=4)

    -- Input
    fName             : String - adress file to read
    nLines            : Number of lines to eliminate at beginning
    nKeep             : Number of lines to keep [Default=None], after nLines eliminated
    nEnd              : Number of lines to eliminate at end [Not functional currently
    fileTemp          : Used as temp file - if not given, created
    """
    
    d = readString(fName)
    d = d[nLines:]
    # print(d)
    # return d
    if nKeep is not None:
        d = d[:nKeep]
    A = convertDataToNumpy(d,fileName=fileTemp, delimiter=delimiter)
    return A

def loadNumpyBlock(fName, nBlock, fileTemp=None, delimiter=None):
    """
    -- fName file 
    # Comment
    Block0
    # Comment
    Block1
    ...
    # Comment
    BlockK

    -- Return Block_$nBlock as numpy array
    """
    d = readString(fName)
    N = len(d)
    ind = [i for i in range(N) if d[i][0] == '#']
    l = []
    n = len(ind)

    # 1 block
    if n==0: return convertDataToNumpy(d,fileName=fileTemp, delimiter=delimiter)

    # Multiple Blocks
    for k in range(len(ind)):
        k1 = ind[k]
        try: k2 = ind[k+1]
        except: k2 = -1
        l.append(convertDataToNumpy(d[k1:k2],fileName=fileTemp, delimiter=delimiter))
    
    return l[nBlock]

def convert_bytes(num):
    
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)

def _getSizeWriteRead(o):
    """Get Size of Object using np.save,load"""
    f = 'obj123456.npy'
    np.save(f,o)
    n= file_size(f)
    os.system('rm %s'%(f))
    return n

# -------------
# Get Memory Size of Python Object
# -------------
def getSize(o,see=None, ):
    """Recursively finds size of objects"""
    try:
        return _getSizeWriteRead(o)
    except:
        print('Fail to obtain size with write/read method')
    def get_size(obj, seen=None):
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        # Important mark as seen *before* entering recursion to gracefully handle
        # self-referential objects
        seen.add(obj_id)
        if isinstance(obj, dict):
            size += sum([get_size(v, seen) for v in obj.values()])
            size += sum([get_size(k, seen) for k in obj.keys()])
        elif hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
        return size
    s = get_size(o, seen=see)
    print('--> %5.2f MB'%(s/100000.))
    return s
# End Function getSize
# --------


# -------------
# Write String To file
# -------------
def writeString(fName,message):
    """
    Write String To file
    """
    try: 
        with open(fName, 'w') as fout:
            fout.writelines(message)
        return 0
    except:
        print('generalFunctions.py: Error Writing in file %s'%(fName))
        return -1
# End Function writeString
# --------


# -------------
# Read String To file
# -------------
def readString(fName):
    """
    Read String from file: list of strings
    """
    with open(fName, 'r', encoding='utf-8') as fin:
        data      = fin.read().splitlines(True)
    return data
# End Function readString
# --------


#--------------------
# Copy Dictionary/Default Dictionary, depth=3
#--------------------
def initDictFull(d,dOld,keep=True):
    """
    d, dOld are dictionaries
    Input:
    -> d       : Incomplete Dictionary (new)
    -> dOld    : Full Default Dictionary
    Output:
    dNew
    For all fields in dOld, if it exists in d, value of d will be used. dOld otherwise
    For fields in d, not belonging to dOld [even deeper]:
        --> they are kept as such [keep=True]
        --> Not Used [keep=False]
    Recursive Depth

    Example
    a      = {'person': {'name': 'Alex', 'age': {'date': 'April86', 'value': 31}}}
    defaut = {'person': {'name': 'Name', 'surname': 'Name', 'age': {'date': 'date', 'value': 0}}, 'passport': None}
    n      = initDictFull(a,defaut)
    n      = {'passport': None,
 'person': {'age': {'date': 'April86', 'value': 31},
  'name': 'Alex',
  'surname': 'Name'}}
    """
    # Initialization
    if type(d) != dict: return d
    dictCase = {}

    # Run dOld
    for args in dOld.keys():
        if args in d:
            dictCase[args]   = initDictFull(d[args],dOld[args])
        else:
            dictCase[args]   = deepcopy(dOld[args])

    # Run d
    if keep:
        for args in d.keys():
            if args not in dOld:  dictCase[args]   = deepcopy(d[args])
    return dictCase
    
# End Function initDictFull
#--------------------


def extractDict(opt,name,default):
    """Extract Information from opt
    opt[name] if exist; default otherwise
    -- Tutorial
    opt = {5:6,7:'lj'}
    extractDict(opt,5,None)   -> 6
    extractDict(opt,'n',None) -> None
    """
    try: return opt[name]
    except: return default

#--------------------
# Initialize Function
#--------------------
def funInit(d,name=None, value=None):
    """
    var = funInit(name='var',value=var,d)

    Example: d = {'pf': {'val': 1, 'color': True}}
    value = {'val': 3}
    var = funInit(d, name='pf', value=value) = {'pf': {'val': 3, 'color': True}}

    -> if var not None, then var = d['var'] completed
    name:  String
    value: float, dict... -> if dict, recursive fields not provided are added from d[name]
    d   : Dictionary of Default Values
    """
    try:
        mem = deepcopy(d[name])
    except:
        mem = name
    if value is None: return mem
    else:
        if type(value) == dict:
            valueNew = initDictFull(value,d[name])
            return valueNew
        return value
    
# End Function funInit
#--------------------

def lineString(name, data,find=False):
    """
    Search indices of appearance of name at the begining of list of Strings
    Input: 
    -> name     : String
    -> data     : List of String
    -> find     : Bool - If True, seeks for name anywhere in the line
    Output:
    -> l        : data[l[k]] starts with name
    """
    n = len(name)
    if find:
        return [k for k in range(len(data)) if name in data[k]] 
    else:
        return [k for k in range(len(data)) if name in data[k][:n]] 

def modifWordFile(fileName, oldWord, newWord):
    """
    Replace string oldWorld by newWord everywhere if appears in fileName
    --> Save it then

    -- Tutorial
    modifWordFile('myFile.dat', '$ADRESS$', './newFolder/')
    """
    # Search line
    data = readString(fileName)
    ind = lineString(oldWord, data, find=True)

    # Replace, save
    for i in ind: 
        data[i] = data[i].replace(oldWord,newWord)
    writeString(fileName,data)
    return 


    
#--------------------
# Expand dictionary containing options into List of dictionaries
#--------------------
def listDictOptions(d, dInit={}):
    """
    Expand dictionary containing options into List of dictionaries
    --> dInit: any dictionary
    --> d    : dictionary. for values that are list, expansion is done. Otherwise, simple copy

    Ex:
    --> d = {'a': [0, 5], 't': [True, False], 'g': [True, False], 6: []}
    --> listDictionOptions(d) = 
[{'a': 0, 't': True, 'g': True, 6: [], 4: 4, 5: 5},
 {'a': 0, 't': True, 'g': False, 6: [], 4: 4, 5: 5},
 {'a': 0, 't': False, 'g': True, 6: [], 4: 4, 5: 5},
 {'a': 0, 't': False, 'g': False, 6: [], 4: 4, 5: 5},
 {'a': 5, 't': True, 'g': True, 6: [], 4: 4, 5: 5},
 {'a': 5, 't': True, 'g': False, 6: [], 4: 4, 5: 5},
 {'a': 5, 't': False, 'g': True, 6: [], 4: 4, 5: 5},
 {'a': 5, 't': False, 'g': False, 6: [], 4: 4, 5: 5}]
    """
    if d == {}: return [dInit]
    else:
        # Init
        arg = list(d.keys())[0] ; dcp = deepcopy(d) ; del dcp[arg]
        resCp = listDictOptions(dcp,dInit=dInit)

        # Recursion Basic [value is list]
        try:
            v = d[arg][0]
            return [{**{arg:v},**resCp[k]} for v in d[arg] for k in range(len(resCp))  ] 

        # Recursion Easy [value not list --> copy as such]
        except:
            return [{**{arg:d[arg]},**resCp[k]} for k in range(len(resCp))  ] 
# End Function listDictOptions
#--------------------



# END -------------PATH-------------------------


# -----------------SAMPLING---------------------
def mic_evaluate(xv,fv,method='default', **opt):
    """
    -- MIC Evaluation "Mutual Information Criterion" using minepy (default parameters)
    mic = mic_evaluation(xv,fv)

    -- Input
    xv             : (N,d) array datas
    fv             : (N,)  array values
    **opt          : alpha=,c= for minepy.MINE

    -- Output
    mic            : (d,) array - normalized values -> mic.sum()=1
    mic[i] = "influence i-th dimension of fv"
    """
    # INitialization
    (N,d) = xv.shape
    assert N==len(fv)

    # i-th variable
    def f(i):
        M = minepy.MINE(**opt)
        M.compute_score(xv[:,i],fv)
        return M.mic()

    # Output
    mic = np.array([f(k) for k in range(d)])
    return mic/mic.sum()
    
#--------------------
# Compute N LHS samples in [xmin,xmax], multidimensional
#--------------------
# Call x = sampleLHS(N, xmin, xmax)
def sampleLHS(N0, xmin, xmax, criterion=None):
    """
    Input:
    -> N               : Integer
    -> xmin            : (d,) array or float
    -> xmax            : (d,) array or float
    Output:
    -> x               : (N,d) array
    """
    # Criterion: maximin, center, centermaximin, correlation
    if criterion is None: criterion='maximin'
    try:
        d = len(xmin)
    except:
        xmin = [xmin] ; xmax = [xmax]
    try: a = xmin-xmax
    except:
        xmin = np.array(xmin) ; xmax = np.array(xmax)

    # Scaling Matrices 
    N                  = int(N0)
    d                  = len(xmin)
    M_coef             = np.diag(xmax-xmin).reshape(d,d)       # dxd Matrix
    M_add              = (np.ones((N,d))*xmin).reshape(N,d)    # Nxd Matrix
    # Case N = 0
    if N == 0:
        return np.zeros(0).reshape(0,d)
    
    # LHS Sampling : N points in dimension d  
    uniPointsNorm      = lhs(d, samples=N, criterion=criterion)
    x                  = np.dot(uniPointsNorm,M_coef) + M_add

    # Output
    return x

# end function sampleLHS
#--------------------


def spaceAS_bounds(xmin=None, xmax=None, V=None, nMC=1e6,factor=2):

    """
    -- Estimate zmin,zmax in AS Subspace -- Not Optimal So far
    zmin,zmax = spaceAS_bounds(xmin=xmin, xmax=xmax, V=V, nMC=1e6, show=show)

    -- Input
    xmin,xmax            : (d,) array - Bounds of Initial Design Space
    nMC                  : Number of MC used to sample x, and estimate zmin,zmax (if not provided)
    V                    : (p,d) array - V[k,:] is k-th eigenvector, AS space
    factor               : in ]1,2] : diminuishing zmin,zmax space

    -- Output
    zmin, zmax           : (p,) Bounds
    """
    
    # Sanity Check
    assert xmin is not None
    assert xmax is not None
    d=len(xmin)
    assert V is not None
    assert V.shape[1]==d
    N = int(nMC)
    p = V.shape[0]
    
    # x Sampling
    x = np.ones((N,d))
    for k in range(d):
        x[:,k] =  np.random.uniform(xmin[k],xmax[k],N)

    # z Space
    z = np.ones((N,p))
    for k in range(p):
        z[:,k] = np.dot(x,V[k,:])
    zmin = np.min(z,axis=0)
    zmax = np.max(z,axis=0)

    # Regularization
    zC   = (zmin+zmax)*0.5
    dz   = (zmax-zmin)*0.5
    zmin = zC-dz/factor
    zmax = zC+dz/factor
    return zmin,zmax


def sampleUniform(N,xmin,xmax):
    """
    N = integer
    xmin,xmax = (d,) arrays
    Return (N,d) array of uniform samples
    """
    d = len(xmin)
    N = int(N)
    x = np.ones((N,d))
    for k in range(d):
        x[:,k] =  np.random.uniform(xmin[k],xmax[k],N)
    return x


def sampleLHS_spaceAS(N0, xmin=None, xmax=None, zmin=None, zmax=None, nMC=1e6, V=None, kLHS=10, criterion=None):
    """
    -- Sample LHS in AS Subspace -- Not Optimal So far
    _,_, zLHS, xLHS = sampleLHS_spaceAS(N0, xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, V=V, kLHS=10)
    or
    zmin,zmax, zLHS, xLHS = sampleLHS_spaceAS(N0, xmin=xmin, xmax=xmax, V=V, kLHS=10)
    
    -- Input
    N0                   : Number of LHS points required
    xmin,xmax            : (d,) array - Bounds of Initial Design Space
    zmin,zmax            : (p,) array - bounds of AS Space -- used for LHS
    nMC                  : Number of MC used to sample x, and estimate zmin,zmax (if not provided)
    V                    : (p,d) array - V[k,:] is k-th eigenvector, AS space
    kLHS                 : kLHS*N0 LHS in [zmin,zmax] are generated. The ones outside [xmin,xmax] are removed. N0 among others are selected randomly. kLHS automatically increased if not enough

    -- Output
    zmin, zmax           : (p,) Bounds
    zLHS                 : (N0,p) LHS samples in AS space
    xLHS                 : (N0,d) LHS samples in original space
    """
    # Sanity Check
    assert xmin is not None
    assert xmax is not None
    d=len(xmin)
    assert V is not None
    assert V.shape[1]==d
    N = int(nMC)
    p = V.shape[0]
    
    # zmin, zmax Evaluation
    if zmin is None or zmax is None:
        # x Sampling
        x = np.ones((N,d))
        for k in range(d):
            x[:,k] =  np.random.uniform(xmin[k],xmax[k],N)

        # z Space
        z = np.ones((N,p))
        for k in range(p):
            z[:,k] = np.dot(x,V[k,:])
        zmin = np.min(z,axis=0)
        zmax = np.max(z,axis=0)

    # Sample LHS
    zLHS = sampleLHS(N0*kLHS, zmin, zmax, criterion=criterion)


    # z = np.dot(x,V.T)
    
    # Convert to Design Space, take-off off design points
    xLHS = np.dot(zLHS,V)

    def test(x):
        mem = True
        for k in range(d):
            if x[k]>xmax[k] or x[k]<xmin[k]: mem=False
        return mem
    
    ind = np.array([i for i in range(xLHS.shape[0]) if test(xLHS[i,:])])
    if len(ind)==0:
        xLHS = np.zeros((0,d))
    else: 
        xLHS = xLHS[ind,:]
        zLHS = zLHS[ind,:]

    # Not enough Points --> Increase number of values
    if xLHS.shape[0] < N0:
        print('Failed LHS: %s/%s [Total LHS=%s] completed so far'%(xLHS.shape[0],N0, N0*kLHS))
        return sampleLHS_spaceAS(N0, xmin=xmin, xmax=xmax, zmin=zmin, zmax=zmax, nMC=nMC, V=V, kLHS=2*kLHS, criterion=None)
    else:
        ind0, _, _ = randomInteger(ni=N0,ntot=xLHS.shape[0])
        return zmin,zmax, zLHS[ind0,:], xLHS[ind0,:]
    
    
        
    

#--------------------
# Test if a set of points p, is in the convex hull defined by the set of points cloud
#--------------------
# Call: (ind, p_in, p_out) = which_inConvexHull(p, cloud)
def which_inConvexHull(p, cloud):
    """
    Input: 
     -> p              : (N,d) array
     -> cloud          : (M,d) array
    Output: 
     -> ind            : (N,) array - indices of points p[ind,:] such that are in the convex hull of cloud
     -> p_in           : (q,d) array - points p in hull
     -> p_out          : (N-q,d) array - points p out of hull
    """
    # Construct Delaunay Mesh of cloud
    if not isinstance(cloud,Delaunay):
        cloud              = Delaunay(cloud)

    # Find Indices in Hull, and construct points
    ind      = cloud.find_simplex(p)>=0
    indInv   = np.logical_not(ind)
    p_in     = np.copy(p[ind,:])
    p_out    = np.copy(p[indInv,:])

    # Output
    return (ind, p_in, p_out)
# End function which_inConvexHull
#--------------------


# -------------
# Weighted K-Means: 
# -------------
def weightedKmeans(x=None,K=None, TOL = 1e-4, NMAX = 100, f = None, optFun={}, regular=False):
    """
    Weighted K-means: Very Similar to Regular K-means. When Updating the centroids, a weighted mean on belonging samples is performed, throught f function (if None, uniform weight used).

    -> wk   = f(xk)
    -> mu_j = (sum_k wk.xk)/(sum_k wk)

    Call: res = weightedKmeans(x=x,K=K, TOL = 1e-4, NMAX = 100, f = fWeight)
    pred = res['pred'], mu = res['mu']

    Input: 
     -> x      : (n,d) array, n points in d dimensions
     -> K      : integer, number of Centroids
     -> TOL    : float - Convergence Criterion on Centroids 
     -> NMAX   : Maximum Number of loop
     -> f      : (n,d) array -> (n,) weight function 
     -> optFun : dictionary - options for f: f(x,**optFun)
     -> regular: if True, then regular K-means is used

    Output:
     -> success: True if err < TOL
     -> pred   : (n,) array, with values in [|0,K-1|]. xk = x[pred==k,:] is the set of points in cluster k, whose centroid is mu[k,:]
     -> centroid: (K,d) array, K centroids
     -> xPred  : list of K (nj,d) arrays, sum_nj=n. xPred[j] is the set of points belonging to group j, mu[j,:] being the centroid
     -> nIter  : Number of Iterations
     -> time   : Elapsed Time (s)
    """
    # ------
    # Initialization: Initialize Centroid Randomly
    # ------
    # General
    assert x is not None
    assert K is not None
    t0 = time.time()
    k = 0
    err = 100
    (n,d) = x.shape

    # f
    if f is None:
        def fx(xk):
            return 1.*np.ones(xk.shape[0])
    else:
        def fx(xk): return f(xk,**optFun)

    # Function Compute Centroid From set of points x
    def centFun(xk):
        fxk = fx(xk) ; (nk,dk) = xk.shape 
        w_f = np.sum(fxk)
        yk = np.repeat(fxk,dk).reshape(nk,dk)
        yk = yk*xk
        return np.sum(yk,axis=0)/w_f
    def compCentroid(xAll,predAll):
        muAll = np.zeros((K,d))
        for i in range(K):
            muAll[i,:] = centFun(xAll[predAll==i,:])
        return muAll

    
    # Init Centroid: Regular Centroids
    try: 
        pred = KMeans(n_clusters=K).fit_predict(x)
        if regular:
            xCent    = np.vstack([x[pred == i].mean(0) for i in range(pred.max() + 1)])
            xCent    = xCent.reshape(xCent.size/d,d)
            out = {'success': True, 'centroid': xCent , 'nIter': 0, 'time': time.time()-t0 , 'pred': pred , 'xPred': [x[pred==j,:] for j in range(K)]}
            return out
    except:
        print('--> Weighted K-Means Algorithm: Regular K-Means[scikit-learn] not available')
        print('--> Random Centroids Initialization')
        pred = np.random.randint(K, size=n)
        if regular: print('--> K-Means Algorithm: Regular K-Means[scikit-learn] not available, so regular algorithm in-House used.')

    mu = compCentroid(x,pred)
    muNew = np.copy(mu)
    
    # Loop
    while (k < NMAX) and (err > TOL):

        # Cluster Assignment: Prediction Cluster for all samples
        """
        for j in range(n):
            dist = np.hstack([np.linalg.norm(x[j,:] - mu[i,:]) for i in range(K)])
            pred[j] = np.argmin(dist)
        """
        dist = distance.cdist(x,mu)         # ||x_i-mu_j||
        pred = np.argmin(dist,axis = 1)     # Argmin_j ||x_i-mu_j||, for all i in [0,n-1]
        
        # Update Centroid
        for i in range(K):
            muNew[i,:] = centFun(x[pred==i,:])

        # Convergence Criterion
        err = 0
        for i in range(K):
            err += np.linalg.norm(muNew[i,:] - mu[i,:])

        # Update Loop
        k += 1
        mu = np.copy(muNew)

    # Output
    xPred = [x[pred==j,:] for j in range(K)]
    out = {'success': err < TOL, 'centroid': mu, 'nIter': k, 'time': time.time()-t0 , 'pred': pred , 'xPred': xPred}
    return out
# End function weightedKmeans
# -------------

def kMeansCenters(u, K, outputData=False) : 
    """
    Compute K Centers from a set of points u, using kMeans algo (Scikit Learn)
    -- Call: 
    (uCent, y_pred) = kMeansCenters(u, K)
    (uData, ind   ) = kMeansCenters(u, K, outputData=True)
    Input : 
    -> u                : (N,d) array of points
    -> K                : Number of clusters supposed
    -> outputData       : If True, approximation of centroids from datas is provided
    Output : 
    -> center           : (K,d) array containing the k_centers centers of the k_centers clusters found
    -> y_pred           : (N,) array: y_pred in [0,K-1]. Prediction of each region
    u[y_pred = k,:] are points in region k, k in np.arange(K)

    or 
    ind                 : (K,) array of integers - Indices of "centroids" approximations
    uData               : (K,d) array - Centroids belonging to dataset u
    """
    # Check Number of elements
    (N,d)    = u.shape
    if (N <= K):
        print("In kMeansCenters, less points in u = ", N," than clusters K_clusters = ", K)
        print("Return u and empty y_pred")
        y_pred    = np.zeros(0)
        if outputData == False: return (u, y_pred)
        else:                   return (u,np.arange(N))

    # Init Clusters object
    y_pred   = KMeans(n_clusters = K).fit_predict(u)
    uCent    = np.vstack([u[y_pred == i].mean(0) for i in range(y_pred.max() + 1)])
    uCent    = uCent.reshape(int(uCent.size/d),d)

    # Centroid, y_pred
    if outputData == False:     return (uCent, y_pred)
    
    # Centroids -> Approximation From Samples
    uk = []
    for i in range(K):
        ind = np.array([k for k in range(N) if y_pred[k]==i])
        c = u[ind,:].mean(0)
        dist = np.array([np.linalg.norm(u[k,:]-c) for k in ind])
        i0 = np.argsort(dist)[0]
        uk.append(u[ind[i0],:])
    uk = np.vstack(uk)

    ind = findArrayIndice(u,uk)
    return uk, ind


def findArrayIndice(xOld,x):
    """
    xOld  = (N,d) array
    x     = (n,d) n<=N
    Find ind = (n,) Integers, s.t.
    --> x = xOld[ind,:]
    """
    # Initialization
    if len(x.shape)==1: x = x.reshape(1,-1)
    if len(xOld.shape)==1: xOld = xOld.reshape(1,-1)
    N = xOld.shape[0]
    n = x.shape[0]
    ind = []

    # Find
    for i in range(n):
        try: k = [p for p in range(N) if np.allclose(x[i,:],xOld[p,:])][0]
        except: continue
        ind.append(k)
    ind = np.array(ind)
    # ind.sort()
    return ind


def chooseKpoints(x=None, K=None, xv=None, TOL=None, returnInd=False):
    """
    Selection of K samples among x, where xv is the current DoE 
    --> Kmeans to obtain K classes [see pred]
    --> For each class, the sample selected is the one the further from the DoE[if given]
                                               the closest to the cluster mean

    -- Input: 
    x                  : (nx,d) array 
    K                  : Integer - Number of Classes. If
    xv                 : Optional (nv,d) array -- Represents the DoE. In case several points belong to the same class, the one the furthest from the DoE is taken
    TOL                : |xi-xj|^2 > TOL only
    returnInd          : If True, returns indices of selected points based on x
    -- Output:
    xSel               : (nK,d) -- nK =K normally, but might nK<K (too close points, not enough classes
    ind (only if returnInd =True): xSel,ind = chooseKpoints(x=,K=,returnInd=True)
    """
    # Basic: K too big
    assert K is not None
    (nx,d) = x.shape
    if K >= nx: return x

    # Init solution
    xx = []

    # Kmeans Clustering
    mu, pred = kMeansCenters(x,K)
    
    # Selection according to dist
    for i in np.unique(pred):
        xi = x[pred==i,:]
        mu_i = mu[i,:].reshape(1,-1)
        
        # xv=None: Closest to Mean
        if xv is None:
            r2 = cdist(xi,mu_i).reshape(-1)
            iSel = np.argmin(r2)
            xx.append(xi[iSel,:])
            continue

        # Further from xv
        else:
            r2 = cdist(xi,xv)   ; np.fill_diagonal(r2,r2.max())
            lDist = np.array([r2[k,:].min() for k in range(xi.shape[0])])
            iSel  = np.argmax(lDist)
            xx.append(xi[iSel,:])
    xSel = np.vstack(xx)

    # Output
    if returnInd==False:    return xSel
    return xSel, findArrayIndice(x,xSel)

def filterPoints(x,TOL=None):
    """
    Filter Points: Among x, Too close points to each other are removed
    --> np.inf, nan
    --> |xi-xj|_1 > TOL_total to be valid
    Returns as well the set of kept indices
    Ex: TOL = 1e-4
    xNew,indNew = self.filterPoints(x,TOL=1e-4)
    Input: 
    --> x             : (N,d) array
    --> TOL           : float > 0 for d=1. TOL_total = d*TOL 
    Output: 
    --> xNew          : (P,d) array, 1<=P<= N
    --> indNew        : (P,) Interger array -> x[indNew,:] = xNew
    """
    # Default Values - filter nan, inf
    assert TOL is not None
    N = x.shape[0] ; iDel = [] ; ind = np.arange(N)
    d = x.shape[1]
    
    # Filter nan, inf
    for i in range(N):
        xi = x[i,:].copy()
        if np.isnan(xi).sum()+np.isinf(xi).sum() > 0: iDel.append(i)
    if len(iDel) >0: iDel = np.unique(np.array(iDel))
    indF = np.delete(ind,iDel)
    x    = x[indF,:]

    # Initialization
    xcp            = np.copy(x)
    (N,_)          = x.shape
    ind            = np.arange(N)
    iDel           = []

    # Passage In Standard Space - |xi-xj|, then transform as boolean --> Norm 1
    xij = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            xij[i,j] = np.linalg.norm(x[i,:]-x[j,:],1)
    # xij            = cdist(x,x)
    xij            = xij > TOL*d

    # Research False, and store redondant entries
    for i in range(N):
        for j in range(i+1,N):
            if xij[i,j]: continue
            iDel.append(j)
    if len(iDel) >0: iDel = np.unique(np.array(iDel))

    # Indices to keep
    indF = np.delete(ind,iDel)

    # Delete Elements
    xcp            = xcp[indF,:]
    return xcp,indF
# End Function filterPoints
#--------------------

def randomDiscreteWeight(K,liste,weight, epsilon=1e-6):
    """
    Choose K values Randomly in list liste, with weights weight - Update Iteratively weight, since at each step a weight is chosen, while an element disapears

    Security: each weight lower than epsilon is removed. weight is normalized (sum=1)
    If K<len(liste), then K is set to len(liste), with uniform weights

    -- Input: 
    K               : Integer
    liste           : (n,) list or array (access liste[k])
    weight          : (n,) list or array with values in ]0,1[ -> Normalized
    -- Output:
    out             : list of min(K,n) elements of liste (order counts), unique
    """
    p = len(liste)
    assert len(weight)==p

    # Remove datas is weight<epsilon
    ind = [i for i in range(p) if weight[i]>epsilon]

    if len(ind)==0: return []
    weight = np.array([weight[ind[i]] for i in range(len(ind))])
    weight = weight/(weight.sum())
    liste  = [liste[i] for i in range(len(ind))]

    # K
    if K > len(liste):
        K = len(liste)
        weight = np.ones(K)/(K+0.)

    # Extraction
    f = []
    mem = False
    custm  = ss.rv_discrete(name='custm', values=(np.arange(len(liste)), weight))
    while mem==False:
        i = custm.rvs(size=1)[0]
        if liste[i] not in f: f.append(liste[i])
        if len(f) == min(K,len(liste)): mem=True
    return f

def addRandom_one(xv=None, x=None, xmin=0, xmax=1, N=1e4):
    """
    Choose Randomly 1 point Maximing minimal distance among xv: 
    Max_j Min_i ||xv_i - x_j||

    x = addRandom_one(xv=,xmin=-2,xmax=4,N=1e5)

    -- Input
    xv           : (nv,d) array
    x            : (nx,d) array -- Choice points --if not given, U(xmin,xmax,N)
    xmin,xmax    : (d,) 
    N            : Integer

    -- Output
    (d,) array
    """
    # Initialization
    if len(xv.shape)==1: xv = xv.reshape(-1,1)
    N = int(N) ; d = xv.shape[1]
    if x is None:
        try:
            d = len(xmin)
        except:
            xmin = np.ones(d)*xmin ; xmax=np.ones(d)*xmax
        x = np.vstack([np.random.uniform(xmin[k],xmax[k],N) for k in range(d)]).T
    if len(x.shape)==1: x = x.reshape(-1,1)
    
    # Evaluation
    d = cdist(x,xv)
    dmin = d.min(axis=1)
    ix = dmin.argmax()
    return x[ix,:]

def addRandom(xv=None,K=10,**opt):
    """
    -- Generate K new samples "far" from xv
    x = addRandom(xv=,K=10,xmin=-2,xmax=4,N=1e5)
    
    -- Input
    xv       : (nv,d) array
    K        : Number of Elements to add
    **opt    : see addRandom_one
    -- Output
    x        : (K,d) array
    """
    XV = xv.copy() ; x=[]
    for i in range(K):
        a = addRandom_one(xv=XV,**opt)
        x.append(a)
        XV = concatRow((XV,a))
    return np.vstack(x)

def randomKfold(n,K):
    """
    -- K-fold indices from list of n elements, decomposed in around K sets
    ind,ind_ = randomKfold(n,K)
    x[ind[i],:]  = xi
    x[ind_[i],:] = x without xi

    -- Output
    ind,ind_   : List of K arrays of integers
    """
    # Basic
    l = np.arange(n)
    import random
    random.shuffle(l)
    nn = int(n/K)
    ind = [] ; ind_ = []

    # K sets: ind
    for k in range(K):
        i = l[k*nn:(k+1)*nn]
        ind.append(list(i))
    D = n-nn*K
    for k in range(D):
        ind[k].append(l[K*nn+k])

    # Remove 1D: ind_
    ind_ = []
    for k in range(K):
        i = delArray1D(np.arange(n),ind[k])
        ind_.append(i)
    ind = [np.array(ind[i]) for i in range(K)]
    return ind,ind_


# def randomDiscreteWeight(K,liste,weight):
#     """
#     Choose K values Randomly in list liste, with weights weight - Update Iteratively weight, since at each step a weight is chosen, while an element disapears
#     -- Input: 
#     K               : Integer
#     liste           : (n,) list or array (access liste[k])
#     weight          : (n,) list or array with values in ]0,1[
#     -- Output:
#     out             : list of min(K,n) elements of liste (order counts), unique

#     -- Example
#     w = np.arange(10) ; w = w/np.sum(w) ; l = np.arange(10)
#     randomDiscreteWeight(5,l,w)

#     """
#     if K==1 or len(liste)==1: return liste
#     if K==0 or len(liste)==0: return []
#     print(weight)
#     custm  = ss.rv_discrete(name='custm', values=(np.arange(len(liste)), weight))

#     a = custm.rvs(size=1)[0]
#     b = weight.copy() ; b = np.delete(b,a) ; b = b/np.sum(b)
#     k = list(liste) ; del k[a]
#     n = [a]
#     n.extend(randomDiscreteWeight(K-1,k,b))
#     return n




# -------------
# Cumulative distances: Not Normalized
# -------------
def cumDist(M):
    """
    cumDist(M)
    # Cumulative distances: Not Normalized
    # M.shape = (N,2) -> compute dM[i] = ||M_i - M_i-1||, where M_i = M[i,:]
    # Then, use cumsum to obtain cumulative sum : np.cumsum(dM)

    Call: t = cumDist(M)
    Input: 
    -> M       : (N,d) array
    Output:
    -> t       : (N,)  array. t[0] = 0, t[-1] = length curve defined by points M
    """

    (N,d)      = M.shape           # Points M_i
    Mroll      = np.roll(M,d)      # Points M_i-1, except for indice 0
    subM       = M - Mroll         # Points M_i - M_i-1, i in [1,N-1]
    dM         = np.sum(np.abs(subM)**2,axis=-1)**(0.5)
    dM[0]      = 0                 # Local distance, careful in 0
    return  np.cumsum(dM)
# End Function cumDist


# -------------
# Cumulative distances Normalized : curvilinear abscissa
# -------------
# M.shape = (N,2) -> compute dM[i] = ||M_i - M_i-1||, where M_i = M[i,:]
# Then, use cumsum to obtain cumulative sum : np.cumsum(dM) and normalize
def normCumDist(M):
    """
    normCumDist(M)
    # Cumulative distances Normalized : curvilinear abscissa
    # M.shape = (N,2) -> compute dM[i] = ||M_i - M_i-1||, where M_i = M[i,:]
    # Then, use cumsum to obtain cumulative sum : np.cumsum(dM) and normalize
    Call: t = normCumDist(M)
    
    Input: 
    -> M       : (N,d) array
    Output:
    -> t       : (N,)  array. t[0] = 0, t[-1] = 1
    """
    dM          = cumDist(M)
    return  dM/dM[-1]
# End Function cumDistNorm
#--------------------
# END--------------SAMPLING---------------------


# -----------------OPTIONS----------------------
#--------------------
# Dictionary Building From Default and **opt
#--------------------
def dictBuildDefault(dictDefault, **opt):
    """
    # dictCase = dictBuildDefault(dictDefault, **opt)
    Ex: d1 = {'a': 5, 'b': 3}, opt = {'a': -10}
    dSol = dictBuildDefault(d1,**opt)
    dSol = {'a': -10, 'b': 3}
    """
    
    dictCase = {}
    for args in dictDefault.keys():
        if args in opt:
            dictCase[args]   = opt[args]
        else:
            dictCase[args]   = dictDefault[args]
    return dictCase
# end function dictBuildDefault
#--------------------


#--------------------
# Dictionary Removing From Default and **opt: optNew = opt\{dictDefault}
#--------------------
# Call: optNew = dictRemoveDefault(dictDefault, **opt)
def dictRemoveDefault(dictDefault, **opt):
    optNew   = {}
    for args in opt.keys():
        if args in dictDefault.keys():
            pass
        else:
            optNew[args]        = opt[args]
    return optNew
# end function dictRemoveDefault
#--------------------


#--------------------
# Dictionary Building From Default and **opt, deep = 2: **opt is a dictionary of dictionaries
#--------------------
# Call dictCase = dictBuildDefault2(dictDefault, **opt)
def dictBuildDefault2(dictDefault, **opt):
    dictCase = {}
    for args in dictDefault.keys():
        if args in opt:
            dictCase[args]   = dictBuildDefault(dictDefault[args], **opt[args])
        else:
            dictCase[args]   = dictDefault[args]
    return dictCase
# end function dictBuildDefault2
#--------------------


#--------------------
# Computes a dictionary containing dictDefault keys (nested depth), with dNew values if available (dictDefault values otherwise)
#--------------------
def dictBuildDepth(baseD, newD):
    """
    Computes a dictionary containing dictDefault keys (nested depth), with dNew values if available (dictDefault values otherwise) - recursive function
    Call: dictCase = dictBuildDepth(dictDefault,dNew)

    Input: 
      -> baseD                          : Basis Dictionary
      -> newD                           : new Dictionary
    Output:
      -> d                              : dictionary containing all keys dictDefault, with values modified if in dNew (nested)
    """
    # Basis: Leaf
    if newD == {}:
        return baseD

    if not(hasattr(newD,'keys')):
        return newD

    # Recursion Function: Loop on arg
    d = {}
    for arg in baseD.keys():
        if arg in newD.keys():
            d.update({arg: dictBuildDepth(baseD[arg], newD[arg])})
        else:
            d[arg] = baseD[arg]
    return d
# end function dictBuildDepth
#--------------------


#--------------------
# Dictionary Inversion: key <-> value 
#--------------------
# Call newDict = inverseDictEntry(dictIn)
def inverseDictionary(dictIn):
    newDict = {}
    for key, value in dictIn.items():
        newDict[value] = key
    return newDict
# End Function inverseDictionary


#--------------------
# Using list as keys for dictionary
#--------------------
# Call: newDict = dictList(d,l). Ex: d = {1:1, 2:2, 3:3, 4:4} ; l = [1,3] ; newDict = {1:1,3:3}
def dictList(d,l):
    newDict = {}
    for i in range(len(l)):
        key = l[i]
        if key in d.keys():
            newDict[key] = d[key]
    return newDict
# End Function dictList


#--------------------
# Using list as keys for dictionary output
#--------------------
# Call: val = dictListValue(d,l). Ex: d = {1:1, 2:2, 3:3, 4:4} ; l = [1,3] ; val = [1,3]
def dictListValue(d,l):
    val = []
    for i in range(len(l)):
        key = l[i]
        if key in d.keys():
            val.append(d[key])
    return val
# End Function dictListValue


#--------------------
# Function to extend list string
#--------------------
def extendListString(l):
    m = ''
    for ll in l:  m += str(ll)
    return m
# End Function extendListString

def cleanDictKey(name,opt,defaultValue=None):
    """
    Replace all entries corresponding to key $name by $defaultValue - recursive function
    """
    o = {}
    if type(opt) is not dict: return opt
    for k,v in opt.items():
        if k == name: o[k] = defaultValue
        else: o[k] = cleanDictKey(name,v)
    return o


#--------------------
# Words extraction between separators
#--------------------
def extractWord(name, separator='"'):
    """
    ['lj','mlkj','r'] = extractWord('"lj""mlkj","r"', separator='"')
    -> Extract list of words contained between two separators
    """
    # Init
    n = len(name)
    l = []
    temp = ''
    i = 0

    # Loop
    while i < n:
        if name[i] == separator:
            j = i+1
            while (j < n):
                if name[j] == separator:
                    temp = name[i+1:j]
                    l.append(temp)
                    break
                j +=1
            i = j
        i +=1
    return l
# End Function extractWord
                                                      


#--------------------
# 1D Numpy array Inversion: ind[indInv] = range
#--------------------
def inverseArray(a):
    """
    inverseArray(a):
    # 1D Numpy array Inversion: ind[indInv] = range
    a   : (n,) array, with {a[i]} = range(n)
    b = inverseArray(a), (n,) array
    --> a[b] = range(n)
    """
    assert len(a.shape) == 1, 'Only for 1D array'
    return np.array([a[i] for i in a])
# End Function inverseDictionary


#--------------------
# Function for Selection of ni values among ntot
#--------------------
def randomInteger(ni=None,ntot=None,seed=None, xv=None, fv=None, fullOutput=False):
    """
    Function for Selection of ni values among ntot
    ind,_,_     = randomInteger(ni=5, ntot=20)
    ind, xi, fi = randomInteger(ni=5, ntot=20, xv=xv, fv=fv)
    ind, xv, fv, xx, fx =  randomInteger(ni=5, ntot=20, xv=xAll, fv=fAll, fullOutput=True)


    -- Input
    ni           : Number of values to select
    ntot         : Total Number of Values [if xv, fv not given]
    seed         : integer [optional]
    xv           : (ntot,d) array
    fv           : (ntot,)  array
    fullOutput   : Bool - If True, returns also xx, fx
    
    -- Output
    ind          : (ni,) array integers
    xv, fv       : (ni,d), (ni,) arrays 
    xx, fx       : (ntot-ni,d), (ntot-ni,) arrays
    """

    import random
    if seed is not None: np.random.seed(seed)
    
    # Initialization
    if fv is not None: ntot = len(fv)
    ind    = range(ntot)
    indv   = np.array(random.sample(range(ntot), ni))
    indx   = delArray1D(ind,indv)
    try:
        xA = xv.copy() ; fA = fv.copy()
    except:
        xA = xv ; fA = fv
        
    # if xv, fv not None
    if xv is not None:   xv = xA[indv,:].copy()
    if fv is not None:   fv = fA[indv].copy()

    # Output
    if fullOutput == False:
        return indv,xv,fv
    else:
        return indv,xv,fv, xA[indx,:].copy(), fA[indx].copy()
# End Function randomInteger
#--------------------


#--------------------
# Function for Selection Next Color
#--------------------
def selectColor(lColor=[],i=0):
    """
    Function for Selection Next Color
    -> if   colorList[i] exists, fine
    -> else colorList[i] set as new color not existing
    Input: 
    -> lColor    : List of Strings
    -> integer
    Output:
    -> newColor  : Updated List of colors such that until i, colorList is Set
    """
    # Easy Case
    n = len(lColor)
    if n>i: return lColor
    nTot = len(colorList)
    
    # Set of Colors
    for ki in range(n,i):
        for k in range(nTot):
            a = colorList[k]
            if a in lColor: continue
            else: break
        lColor.append(a)
                                                                            
    return lColor
# End Function selectColor
#--------------------


# --------
# Concatenate 1D arrays, in One block
# --------
# Call xv = concat1D(x,y,...,z) ; xv = [x,y,...,z]
def concat1D(l):
    """
    Input:
    -> l                  : list of d (N,) arrays
    Output:
    -> tab                : (N,d) array
    """
    return np.vstack(l).T
# End function concat1D


# --------
# Concatenate in Columns
# --------
def concatColumns(l):
    """
    Concatenate in Columns
    Input:
    -> l                  : list of (N,dk) or (N,) or None arrays
    Output:
    -> tab                : (N,d) array, where d=sum dk
    """
    m=[]
    n=len(l)
    for i in range(n):
        li=l[i]
        if li is None: continue
        if len(li.shape)==1: li=li.reshape(-1,1)
        m.append(li)
    return np.hstack(m)
# End function concat1D


# --------
# Repeat 1D array n rows
# --------
def repeat1D(A,n):
    """
    # Repeat 1D array n rows
    x = repeat1D(A,n)
    Input:
    -> A                  : (N,) array or (1,N) array
    Output:               : (n,N) array
    """
    try:
        if len(A.shape)==2: A = A.reshape(-1)
    except: pass
    return np.repeat(A,n).reshape(len(A),n).T
# End function repeat1D


# --------
# Concatenate d-D arrays, in rows
# --------
def concatRow(l):
    """
Call xv = concatRow((x,y,...,z)) ; xv = [ x ]
                                      | y |
                                      |...|
                                      [ z ]
    Input:
    -> l                  : list of (Ni,d) arrays -> (1,d) at least!! or None
    Output:
    -> tab                : (N,d) array, N = sum Ni [return np.zeros((0,0)) if all None
    """
    lNew=[]
    for i in range(len(l)):
        x = l[i]
        if x is None: continue
        if len(x.shape)==1: x = x.reshape(1,-1)
        lNew.append(x)
    if lNew == []: return np.zeros((0,0))
    return np.concatenate(lNew, axis = 0)
# End function concatRow

# Find indices ind such that b[ind] in a 
def findArray(a,b):
    """
    Careful: Double loop -> Very Slow if len(a) > 2000
    Find indices ind such that b[ind] in a
    ->  b[ind] = a if a is totally included in b
    Example: a = [2,8,9],    b = [9,1,2,7,8] -> c = findArray(a,b)
                          b[c] = [2,8,9]        c = [0,2,4]

    Input: 
    -> a         : (n,) array
    -> b         : (p,) array

    Output:
    -> c         : (k,) array, k <= n
    """
    return np.array([j for i in range(len(a)) for j in range(len(b))  if a[i] == b[j]])
# End function findArray


# Function Computing ||x-y||^2 = [ |xi-yj|^2 ]_i,j matrix
def cdist(x,y, enforce = False):
    """
    --> NOW: Uses distance.cdist(x,y), so |x-y| is given
    Function Computing ||x-y|| = [ |xi-yj|
    
    Input: 
    -> x            : (N,d) array
    -> y            : (P,d) array

    Output:
    -> r            : (N,P) array
    """
    return distance.cdist(x,y)  
#  x2                         = np.sum(np.square(x),1)
   #  y2                         = np.sum(np.square(y),1)
   #  r2                         = -2.*np.dot(x,y.T) + x2[:,None] + y2[None,:]
    
   #  # If x = y, enforce diagonal to 0
   #  if enforce:
   #      np.fill_diagonal(r2,0)

   #  # Ensure r2 >= 0
   #  return r2.clip(min=0)
    
# End Function cdist

# --------
# Create Dictionary - column indexing
# --------
def columnIndexing(name,value):
    """
    Create Dictionary - column indexing 
    --> Call: 
    ind_col, ntot = columnIndexing(['pf','gamma','k'],[5,1,[2,5]])
    where ntot = 5+1+2*5
    ind_col = {'pf':[0,1,2,3,4], 'gamma': [5], 'k': [[6,7,8,9,10],[11,12,13,14,15]]}

    ind_col['pf']   = ...
    ind_col['k'][0] = ...
    ind_col['k'][1] = ...

    Input:
    -> name               : list of n string
    -> value              : list of n integer/array
    Output:
    -> fold               : idem
    """
    # Case Only values
    def funBasic(nam,valu):
        # Basic Info
        n = len(nam)
        ntot = sum(valu)
        assert len(nam) == len(valu), 'name=%s, value=%s'%(nam,valu)
        valu = [int(valu[i]) for i in range(len(valu))]
        
        # Building Dictionary
        ind = {nam[0]: [j for j in range(valu[0])]}
        for i in range(1,n):
            ind[nam[i]] = [j+ind[nam[i-1]][-1]+1 for j in range(valu[i])]
        return ind, ntot

    # Basic Info
    n = len(name)
    assert len(name) == len(value), 'name=%s, value=%s'%(name,value)

    # Detect Where arrays - Transform all in values
    arr      = {}
    valueVal = []
    for i in range(n):
        try:
            k = value[i][0]*value[i][1]
            a = True
        except:
            k = value[i]
            a = False
        valueVal.append(int(k))
        arr[name[i]] = [a,value[i]]

    # Problem solved for values
    indVal,ntot = funBasic(name,valueVal)

    # Reshape Correctly
    ind = {}
    for k,v in indVal.items():
        if arr[k][0]:  ind[k] = np.array(v).reshape(arr[k][1])
        else:          ind[k] = np.array(v)

    return ind, ntot
# End function columnIndexing
# --------


# --------
# Create Numpy array from dictionary of numpy arrays, indexed by integers
# --------
def dictToNumpy(d):
    """
    Create Numpy array from dictionary of numpy arrays, indexed by integers [make it sorted, see example]. Index is repeated as the first, considering only integer keys
    --> Call: 
    arr = dictToNumpy(d)
    Ex: d = {'REf': [2, 3 ,4], 2: [2, 5,6], 5: [5, 9.89,6.45], 3: [3, 3,4]}
    arr = [[2 5 6],[3 3 4],[5 9.89 6.45]] --> (N,d) array
    """
    # Function Test Num
    def funNum(x):
        try:
            a = int(x)
            return True
        except:
            return False
    
    # Basic Info
    l = np.array([k for k in d.keys() if funNum(k)])
    l = np.sort(l)
    n = len(l)
    if n == 0: return np.zeros(0)
    m = len(d[l[0]])
    arr = np.zeros((n,m))

    # Loop
    for i in range(n):
        arr[i,:] = d[l[i]]
        
    return arr
# End function dictToNumpy
# --------

def partitionInteger(n,NMAX):
    """
    Partioning Task of size n, if maximal value is NMAX
    Example:
    partitionInteger(10,3) -> 
    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]
    """
    if n<= NMAX: return range(n)
    ii = int(n/NMAX)
    ind = [np.arange(k*NMAX,(k+1)*NMAX) for k in range(ii)]
    ind.append(np.arange(ii*NMAX,n))
    return ind


def delArray1D(ori,ind):
    """
    ind = list or integer

    Remove indices ind in ori - Not efficient if big array
    a = delArray(np.arange(10), [4,5])
    a = [0,1,2,3,6,7,8,9]
    """
    try: 
        return np.array([ori[i] for i in range(len(ori)) if i not in ind])
    except:
        return np.array([ori[i] for i in range(len(ori)) if i != ind])
def delArray2D(ori,ind):
    """
    Remove indices ind in ori
    a = delArray(blade, [4,5]), blade = (N,2) array
    a = blade without blade[i,:], i in ind
    
    """
    ind0 = delArray1D(range(ori.shape[0]),ind)
    return ori[ind0,:]


def interp1D(x0,f0):
    """
    -- 1D INterpolation: use of cubic splines
    f  = interp1D(x,y)
    yy = f(xx) for interpolation
    (n,)->(n,) 
    """
    import scipy
    f = scipy.interpolate.interp1d(x0, f0, kind='cubic')
    
    return f


def interp(x0,f0):
    """
    -- Linear/Scattered Interpolation at least 2D
    f  = interp1D(x,y)
    yy = f(xx) for interpolation
    (n,d)->(n,) 
    """
    import scipy.interpolate as si
    linInt = si.LinearNDInterpolator(x0, f0, fill_value=np.nan, rescale=True)
    neaInt = si.NearestNDInterpolator(x0,f0)
    
    # Function Evaluation
    def f(x):
        fLin         = linInt(x)
        fNea         = neaInt(x)
        ind          = np.isnan(fLin)
        fLin[ind]    = fNea[ind]
        return fLin
    return f
                                    

    
# Second Degree
def solveSecondDegree(a,b,c):
    """ 
    Solve ax**2+b*x+c=0
    -- Output
    None       : if No Solutions
    [x1]         : if One solution
    [x1,x2]      : if two solutions
    """
    delta = b**2-4*a*c
    if delta <0: return None
    d = np.sqrt(delta)
    x1 = (-b-d)/2/a
    x2 = (-b+d)/2/a
    if delta==0:return [x1]
    return [x1,x2]


def integrate(x,f):
    """
    -- Integral (x,f(x)) based on values provided. Simple Trapeze formula
    sum (f[i+1]+f[i])/2*(x[i+1]-x[i])

    -- Input
    x, f          : (n,) array
    -- Output
    I             : float
    """
    assert len(x)==len(f)
    n = len(f)
    df = (f+np.roll(f,1))/2. ; df = df[1:]
    dx = x-np.roll(x,1) ; dx = dx[1:]
    # I = 0
    # for i in range(n-1):
    #     I += (f[i+1]+f[i])*(x[i+1]-x[i])/2
    return (df*dx).sum()
    
def boundaryLayerParameters(u=None,y=None, rho=None, fileInput=None, delimiter=',', show=True, offsetCol=0, nLines=None):
    """
    -- Estimation of BL parameters
    delta, dStar, dStar2, theta, H = boundaryLayerParameters(u=, y=, rho=)

    -- Input
    y = Normal Direction[Wall Distance]- (N,) array Wall Distance
    u = Velocity                       - (N,) array
    rho = Density                      - (N,) array
    offsetCol                          - If offset in file. Order=rho,u,y
    *fileInput: [u,y] = np.loadtxt(fileInput,delimiter=',')
    nLines                             - If not None -> loadNumpyLines is used
    -- Output
    delta                  : BL thickness - u(delta)=0.99u0
    dStar*                 : Displacement Thickness 
    dStar2*                : Energy Thickness       
    theta*                 : Momentum Thickness 
    H*                     : Shape Factor

*   Displacement Thickness: the distance by which a surface would have to be moved in the direction parallel to its normal vector towards the reference plane in an inviscid fluid at u0 to give the same flow rate as occurs between the surface and the reference plane in a real fluid: 
int_0_oo (1-rho*u/rho0/u0)

*   The energy thickness, is the distance by which a surface would have to be moved parallel to its normal vector towards the reference plane in an inviscid fluid stream of velocity to give the same total kinetic energy as exists between the surface and the reference plane in a real fluid
int_0_oo (rho*u/rho0/u0)*(1-u**2/u0**2)

*  The momentum thickness,  is the distance by which a surface would have to be moved perpendicular from the reference plane in an inviscid fluid stream of velocity u 0 {\displaystyle u_{0}} u_{0} to give the same total momentum as exists between the surface and the reference plane in a real fluid.
int_0_oo (rho*u/rho0/u0)*(1-u/u0)

* H Factor [Nature of the flow]: The higher the value of H, the stronger the adverse pressure gradient. A high adverse pressure gradient can greatly reduce the Reynolds number at which transition into turbulence may occur. Conventionally, H = 2.59 (Blasius boundary layer) is typical of laminar flows, while H = 1.3 - 1.4 is typical of turbulent flow:
H=delta*/theta
    """
    # Initialization
    if fileInput is not None:
        if nLines is not None:
            x = loadNumpyLines(fileInput,nLines,delimiter=delimiter)
        else:
            x = np.loadtxt(fileInput,delimiter=delimiter)
        rho = x[:,offsetCol+0] ; u = x[:,offsetCol+1] ; y=x[:,offsetCol+2]
    n = len(y)

    if rho is None:
        print('Density not Provided -> Set to 1')
        rho = np.ones(n)
    assert len(u)==n and len(u) == len(rho)
    u0 = u.max()
    rho0 = rho.max()

    # delta
    delta = np.percentile(y,99)

    # Displacement Thickness - massflow rate
    f = 1 - rho*u/rho0/u0
    dStar = integrate(y,f)

    # Energy Thickness 
    f = rho*u/rho0/u0*(1-u**2/u0**2)
    dStar2 = integrate(y,f)
    
    # Momentum Thickness 
    f = rho*u/rho0/u0*(1-u/u0)
    theta = integrate(y,f)

    # H Factor
    H = dStar/theta

    m = '-- BL Estimation:\n'
    m+= 'u0    = %2.7E\n'%(u0)
    m+= 'rho0  = %2.7E\n'%(rho0)
    m+= 'delta = %2.7E   : BL thickness           - u(delta)=0.99u0\n'%(delta)
    m+= 'dStar = %2.7E   : Displacement Thickness - int_0_oo (1-rho*u/rho0/u0)\n'%(dStar)
    m+= 'dStar2= %2.7E   : Energy Thickness       - int_0_oo (rho*u/rho0/u0)*(1-u**2/u0**2)\n'%(dStar2)
    m+= 'theta = %2.7E   : Momentum Thickness     - int_0_oo (rho*u/rho0/u0)*(1-u/u0)\n'%(theta)
    m+= 'H     = %2.7E   : Shape Factor           - [dStar/theta] - 2.4[Blasius Laminar] - 1.3[Turbulent]\n'%(H)

    # show
    if show:
        print(m)
    return delta, dStar, dStar2, theta, H

    
def gramSchmidt(vectors):
    """
    Orthonormalization of list of basis vectors, starting from the first one
    
    new = gramSchmidt(vectors)

    -- Input:
    vectors       : (n,d) array - n basis initial vectors (first is kept) - vectors[i,:] 

    -- Output:
    new           : (p,d) array - new[i,:] = vector i - p<=n (case vectors numerically linearly dependent)
    new[0,:] = vectors[0,:]/norm(vectors[0,:])
    new*new.T = I_p
    """
    # Input: vectors as list
    # if type(vectors) != list:
    #     vectors = vectors.tolist()

    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if np.linalg.norm(w) > 1e-10:
            basis.append(w/np.linalg.norm(w))
            
    return np.array(basis)

def rankGS(V,v):
    """
    Evaluate importance of vector v in Orthonormal Basis V
    r = ||v - Proj_V(v)||, Proj_V(v) = sum_i V[i,:].v V[i,:]
    if r ~ 0, v almost in Span(V). Otherwise, v brings contribution.

    -- Input: 
    V             : (n,d) array - V[i,:] is i-th vector, with <vi,vj>=delta_ij
    v             : (d,)  array -> v is then normalized
    
    -- Output: 
    r             : float in [0,1]
    """
    v = v/np.linalg.norm(v)
    n,d = V.shape
    assert len(v) == d
    pv = 0
    
    for i in range(n):
        pv = pv + np.sum(V[i,:]*v)*V[i,:]
    y = v-pv
    return np.linalg.norm(y)

def rankGS_list(V,W):
    """
    Evaluate importance of vectors in W w.r.t. Orthonormal Basis V
    see rankGS()
    if r ~ 0, v almost in Span(V). Otherwise, v brings contribution.

    -- Input: 
    V             : (n,d) array - V[i,:] is i-th vector, with <vi,vj>=delta_ij
    W             : (m,d) array -> each of W[k,:] vectors will be normalized
    
    -- Output: 
    r             : (m,) array in [0,1]
    """
    return np.array([rankGS(V,W[i,:]) for i in range(W.shape[0])])

def addGS(V,v,eps=1e-4):
    """
    Add Orthogonal vector to V, based on v. Only if norm final vector is above eps
    Vnew = addGS(V,v,eps=1e-3)

    -- Input: 
    V             : (n,d) array - V[i,:] is i-th vector, with <vi,vj>=delta_ij
    v             : (d,)  array -> v is then normalized

    -- Output: 
    W             : (n+1,d) array if accepted. V=(n,d) otherwise
    """
    v = v/np.linalg.norm(v)
    n,d = V.shape
    assert len(v) == d
    pv = 0
    
    for i in range(n):
        pv = pv + np.sum(V[i,:]*v)*V[i,:]
    y = v-pv
    r = np.linalg.norm(y) 
    if r < eps: return V
    y = y/r
    return concatRow((V,y))
    
def performGS(V,W,eps=1e-4):
    """
    Add vectors from w in V, with Gramm-Schimdt Orthonormalization. 
    U = performGS(V,W) 
    U[:n,:]= V

    Iteratively, vectors of W are ranked. The one with highest rank/independence is then added. 

    -- Input: 
    V             : (n,d) array - V[i,:] is i-th vector, with <vi,vj>=delta_ij
    W             : (m,d) array -> each of W[k,:] vectors will be normalized

    -- Output: 
    U             : (p,d) array, p vectors orthonormalized with U[:n,:]= V, p<=n+m
    """
    p = W.shape[0]
    for k in range(p):
        r   = rankGS_list(V,W)
        i   = np.argmax(r)
        V = addGS(V,W[i,:],eps=eps)
        W = np.delete(W,[i],axis=0)
    return V



def rotBlade(bladePoints, theta, center):
    """
    Rotation Blade (O, theta)
    Computes points from bladePoints (numpy array (N,2)) after rotation of angle theta [rad] of center "center" (numpy array (2,))
    newPoints = rotBlade(bladePoints, theta, center)
    Input: 
    -> bladePoints: (N,2) array
    -> theta      : float (angle in radian)
    -> center     : (2,) array
    """
    newPoints       = np.copy(bladePoints)
    c               = np.cos(theta)
    s               = np.sin(theta)
    dx              = bladePoints[:,0] - center[0]
    dy              = bladePoints[:,1] - center[1]
    newPoints[:,0]  = center[0] + c*dx - s*dy
    newPoints[:,1]  = center[1] + s*dx + c*dy
    return newPoints

def plotTangentPerso(A,t,k=1e-3,color='black',lw=2,zorder=0):
    """Normalized tanget t is used - Dirty Function"""
    f = np.zeros((2,2))
    f[0,:] = A
    t = t/np.linalg.norm(t)
    try:        f[1,:] = A+k*t
    except:     f[1,:] = A+k*np.array(t)
    # plt.arrow(A[0],A[1],t[0]*k,t[1]*k,color='black')
    plt.plot(f[:,0],f[:,1],color=color,lw=lw,zorder=zorder)


def ellipseTE(A=None, B=None, m1=None, m2=None,
              nTE=20, includeAB=False, vBig=1e8, vSmall=1e-8, allEllipse=False, show=False, clockwise=False):
    """
    -- Return Ellipse arc passing through A,B (with resp. tangent vectors (1,m1), (1,m2)), with minimum excentricity
    arc = ellipseTE(A=[[A0x,A0y],[A1x,A1y]], B=[[B0x,B0y],[B1x,B1y]], nTE=20, includeAB=False)
    arc = ellipseTE(A=[[A0x,A0y]], B=[[B0x,B0y]], m1=10, m2=-3, nTE=20, includeAB=True) --> includes A and B

    -- Input
    A, B             : list of (2,) array(s). if m1,m2 is None, tA=vect(A0,A1) normalized (idem for tB)
    m1,m2            : Tangent value in A (resp.B), so that tA=(1,m1), tB=(1,m2) tangent vectors.
    nTE              : Number of Points returned
    includeAB        : if True, points returned include A and B
    vBig,vSmall      : Robustness code - m1=0 not valid for instance
    allEllipse       : if True, returns the full Ellipse
    clockwise        : if True, return in clockwise direction. Anti-clockwise otherwise
    show             : Plot

    -- Output
    M                : (nTE,2) array -> From A to B
Contains artificial robustness parameters to ensure mi!=0 and cos theta - mi sin theta !=0
    """
    # Order A,B: B on the left
    mem = False
    if A[0][0]<B[0][0]:
        C = deepcopy(B)
        B = deepcopy(A)
        A = deepcopy(C)
        mem = True
        if m1 is not None:
            c = m2 ; m2=m1 ; m1 = c
    
    # m1, m2
    if m1 is None: 
        tA = np.array([A[1][0]-A[0][0],A[1][1]-A[0][1]])
        if abs(tA[0])<vSmall: m1 = vBig
        else: m1 = tA[1]/tA[0]
    if m2 is None: 
        tB = np.array([B[1][0]-B[0][0],B[1][1]-B[0][1]])
        if abs(tB[0])<vSmall: m2 = vBig
        else: m2 = tB[1]/tB[0]
    if abs(m1) > vBig: m1 = vBig*np.sign(m1)
    if abs(m2) > vBig: m2 = vBig*np.sign(m2)
    # if invDirection:
    #     m1 = -m1
    #     m2 = -m2
    
    # Work in Reference Frame - theta = (x,AB) -> A(alpha,0), B(-alpha,0)
    xA = A[0][0] ; yA = A[0][1]
    xB = B[0][0] ; yB = B[0][1]
    AB = np.sqrt((xB-xA)**2+(yB-yA)**2)
    theta = np.arcsin((yA-yB)/AB)
    alpha = AB*0.5

    # Ref Frame: m1,m2
    if abs(cos(-theta)-m1*sin(-theta))<vSmall:
        m1N = vBig*np.sign(m1*cos(-theta)+sin(-theta))*np.sign(cos(-theta)-m1*sin(-theta))
    else:
        m1N = (m1*cos(-theta)+sin(-theta))/(cos(-theta)-m1*sin(-theta))
    if abs(cos(-theta)-m2*sin(-theta))<vSmall:
        m2N = vBig*np.sign(m2*cos(-theta)+sin(-theta))*np.sign(cos(-theta)-m2*sin(-theta))
    else:
        m2N = (m2*cos(-theta)+sin(-theta))/(cos(-theta)-m2*sin(-theta))
    if abs(m1N) > vBig: m1N = vBig
    if abs(m2N) > vBig: m2N = vBig

    # Ellipse parameters (standard equation): x**2+Bxy+Cy**2+Dx+Ey+F=0
    A = 1
    B = -(m1N+m2N)/(m1N*m2N)
    C = 1+(m1N+m2N)**2/(2*m1N**2*m2N**2)
    D = 0
    E = alpha*(m1N-m2N)/(m1N*m2N)
    F = -alpha**2

    # Ellipse center/angle
    beta = atan((-A+C-sqrt(B**2+(A-C)**2))/B)
    cx   = (2*C*D-B*E)/(B**2-4*A*C)
    cy   = (2*A*E-B*D)/(B**2-4*A*C)

    # Ellipse a, b -> Modif frame ref  xp**2+Cp.yp**2+Dp.xp+Ep.y+Fp=0
    tp = atan(B/(A-C))*0.5
    Ap = A*cos(tp)**2+B*cos(tp)*sin(tp)+C*sin(tp)**2
    Bp = 0
    Cp = A*sin(tp)**2-B*cos(tp)*sin(tp)+C*cos(tp)**2
    Dp = D*cos(tp)+E*sin(tp)
    Ep = -D*sin(tp)+E*cos(tp)
    Fp=F
    a  = sqrt((-4*Fp*Ap*Cp+Cp*Dp**2+Ap*Ep**2)/(4*Ap**2*Cp))
    b  = sqrt((-4*Fp*Ap*Cp+Cp*Dp**2+Ap*Ep**2)/(4*Ap*Cp**2))
    opt = {'C':[cx,cy], 'a':a, 'b':b, 'theta':beta}
    tA = findParam_ellipse(alpha,0,**opt)
    tB = findParam_ellipse(-alpha,0,**opt)

    # Exchange tA,tB if done previously
    if mem:
        c=tA ; tA=tB ; tB=c
    
    # IncludeAB -> nTE
    if includeAB == False: nTE+=2

    # Param Ellipse [two pieces]
    t  = np.linspace(tA,tB,nTE)
    if tA < tB: tA+=2*np.pi
    else:       tB+=2*np.pi
    tR = np.linspace(tA,tB,nTE)
    if allEllipse:t = None
    x  = ellipse_xy(C=[cx,cy], a=a, b=b, theta=beta, t=t)
    xR = ellipse_xy(C=[cx,cy], a=a, b=b, theta=beta, t=tR)
    
    # Plot Reference Ellipse
    if show: 
        plt.scatter(alpha,0,label='Aref',color='blue',s=100,zorder=10, marker='s')
        plt.scatter(-alpha,0,label='Bref',color='blue',s=100,zorder=10, marker='x')
        plt.scatter(x[:,0],x[:,1], label='ellipseRef',color='blue', s=5, zorder=0)
        # plt.scatter(xR[:,0],xR[:,1], label='ellipseRef_rot',color='green', s=5, zorder=0)
        plotTangentPerso([alpha,0],[1,m1N],zorder=3,k=1)
        plotTangentPerso([-alpha,0],[1,m2N],zorder=3,k=1)
    
    # Rotate, translate
    x[:,0] = x[:,0]+xB+alpha
    x[:,1] = x[:,1]+yB
    x = rotBlade(x, theta, [xB,yB]) 
    xR[:,0] = xR[:,0]+xB+alpha
    xR[:,1] = xR[:,1]+yB
    xR = rotBlade(xR, theta, [xB,yB]) 

    # Selection: angle tc = (BA,BC)
    A = x[0,:] ; B = x[-1,:] ; C = x[1,:]
    u = [A[0]-B[0],A[1]-B[1]] ; v = [C[0]-B[0],C[1]-B[1]]
    det = u[0]*v[1]-u[1]*v[0]
    if det <0 and clockwise or det>0 and clockwise==False: pass
    else: x = xR.copy()
    
    # IncludeAB in a clean way
    if includeAB == False: x = x[1:-1,:]
    else:
        x[0,:] = [xA,yA]
        x[-1,:] = [xB,yB]
    
    # Plot Final Ellipse
    if show:
        plt.scatter(xA,yA,label='A',color='red',s=100,zorder=10, marker='s')
        plt.scatter(xB,yB,label='B',color='red',s=100,zorder=10, marker='x')
        plt.scatter(x[:,0],x[:,1], label='ellipse',color='red', s=5, zorder=1, marker='x')
        # plt.scatter(xR[:,0],xR[:,1], label='ellipse_rot',color='black', s=5, zorder=1, marker='x')
        plotTangentPerso([xA,yA],[1,m1],zorder=3,k=1,color='orange')
        plotTangentPerso([xB,yB],[1,m2],zorder=3,k=1,color='orange')

        plt.legend()
        plt.title('alpha=%s; theta=%s degree'%(alpha,theta*180/np.pi))
        plt.show()
        
    return x

# Ellipse Plot
def ellipse_xy(C=[0,0], a=2, b=1, theta=0, t=None, Npts=1000):
    """
    Return (x,y)(t)

    -- Input
    C            : Ellipse Center
    a, b         : minor/major ellipse axis (indifferent)
    theta        : Ellipse angle theta=(Ox,CF)
    t            : parametrization - 
    """
    if t is None: t = np.linspace(0,2*np.pi,Npts)
    try:     n = len(t)
    except:  t = [t]
    n = len(t)
    X = np.zeros((n,2)) ; c = np.cos(theta) ; s = np.sin(theta)
    # X[:,0] = np.cos(theta)*(C[0]+a*np.cos(t))
    # X[:,1] = np.sin(theta)*(C[1]+b*np.sin(t))
    # X[:,0] = (C[0]+a*np.cos(t))
    # X[:,1] = (C[1]+b*np.sin(t))
    x = a*np.cos(t)
    y = b*np.sin(t)
    X[:,0] = C[0] + c*x - s*y
    X[:,1] = C[1] + s*x + c*y
    return X

def findParam_ellipse(X, Y, C=[0,0], a=2, b=1, theta=0):
    """
    Find value t s.t. A(x,y) belonging to the ellipse of parameters C, a, b, theta
    x_ellipse(t) = x
    y_ellipse(t) = y
    -- Input
    x, y           : float or (N,) arrays

    -- Output
    t              : float or (N,) array in [-pi,pi]
    """
    # Circle coordinates
    c = np.cos(theta) ; s = np.sin(theta)
    x = X - C[0]
    y = Y - C[1]
    xA = c*x+s*y
    yA = -s*x+c*y
    x = xA/a
    y = yA/b

    # Trigonometric Circle: find t
    t = np.arccos(x)*np.sign(y)
    return t

def cubicInterp1D(x=[],y=[],dx=[],dy=[], show=False, alphaPlot=0.1):
    """
    -- Cubic 1D Interpolation
    a,f = cubicInterp1D(x=[0,2],y=[1,5],dx=[-1,2],dy=[0,1])
    so f(x) = sum_i=0,3 ai.x^i
    x=np.linspace(0,1,100); px=f(x)
    satisfying f(xi)=yi and f'(dxi)=dyi

    -- Input
    x       : List of Training locations
    y       : List of Training values
    dx      : List of Training locations for derivative
    dy      : List of Training derivatives
    len(x)+len(dx)=4
    len(x)=len(y), len(dx)=len(dy)
    show    : if True, plot of Polynomial

    -- Output
    a       : (4,) array - coefficients
    f       : (n,) -> (n,) function (also for floats)
    """
    # Initialization
    A = np.zeros((4,4)) ; f = np.zeros(4)
    n = len(x) ; m = len(dx)
    assert len(y)==n and len(dy)==m

    # Fill x,y
    for i in range(n):
        A[i,:] = [x[i]**3,x[i]**2,x[i],1]
        f[i]   = y[i]

    # Fill dx,dy
    for i in range(m):
        A[i+n,:] = [3*dx[i]**2,2*dx[i],1,0]
        f[i+n]   = dy[i]

    # Solve a, create f(x)
    a = np.linalg.solve(A,f)[::-1]
    def f(X):
        return a[3]*X**3+a[2]*X**2+a[1]*X+a[0]

    # Plot if show
    xmin=np.inf;xmax=-np.inf
    if show:
        if n>0:
            xmin=min(np.array(x).min(),xmin)
            xmax=max(np.array(x).max(),xmax)
        if m>0:
            xmin=min(np.array(dx).min(),xmin)
            xmax=max(np.array(dx).max(),xmax)
        dX=xmax-xmin
        X = np.linspace(xmin-alphaPlot*dX,xmax+alphaPlot*dX,100)
        Y = f(X)
        plt.plot(X,Y,color='black',label='P(x)')
        if n>0:
            plt.scatter(x,y,marker='o',color='red',label='yi=P(xi)',s=20)
        if m>0:
            dY = f(np.array(dx))
            plt.scatter(dx,dY,marker='o',color='green',label="dyi=P'(xi)",s=20)
        plt.legend()
        plt.show()
    return a, f


# -------------
# Solve X, then return B
# Compute A, b (no BC)
# Modify to take into account BC
# Solve to compute X
# Complete BC, and get B (output)
# -------------
# Call: B = computeB_spline1D(x, f, alpha, beta, periodic)
def computeB_spline1D(x, f, alpha, beta, periodic):
    """
    Input: 
    -> x,f          : (N,) arrays such that f[i] = f(x[i])
    -> alpha        : float - f'(left)  = alpha
    -> beta         : float - f'(right) = beta
    -> periodic     : boolean (if True, periodic conditions are applied)
    Output:
    -> B            : (N,) array - Spline array parameter solution
   """
    # ---------
    # Initialization
    # ---------
    # Basics
    N            = len(x)
    h            = np.diff(x)
    
    # Compute A, b, no BC
    A            = computeMatrixA_spline1D(x,f)
    b            = computeSnd_b_spline1D(x, f)
    
    # ---------
    # Periodic BC
    # ---------
    if periodic: 
        # Update only A
        A[0,-1]    += h[0]/6.0
        A[-1,0]    += h[N-2]/6.0
    # ---------
    # NO - Periodic BC
    # ---------
    else: 
        # Update A, then b
        A[0,0]     += -h[0]/12.0
        A[-1,-1]   += -h[N-2]/12.0
        b[0]       += alpha/2.0 + (f[1] - f[0])/h[0]/2.0
        b[-1]      += -beta/2.0 + (f[N-1] - f[N-2])/h[N-2]/2.0
        
    # ---------
    # Convert A into sparse, and solve for X
    # ---------
    A_sp         = sparse.csr_matrix(A) 
    X            = sparse.linalg.spsolve(A_sp,b)

    # ---------
    # Compute B
    # ---------
    B            = np.zeros(N)
    B[1:-1]      = X
    if periodic:  # Case Periodic
        B[0]         = B[-2]
        B[1]         = B[-1]
    else:         # Case Dirichlet derivative
        B[0]         = -B[1]/2.0 - 3.0*alpha/h[0] + 3.0*(f[1] - f[0])/h[0]**2
        B[-1]        = -B[-2]/2.0 + 3.0*beta/h[N-2] - 3.0*(f[N-1] - f[N-2])/h[N-2]**2

    # Numerical f'(right)
    betaNum      = (f[-1] - f[-2])/(x[-1] - x[-2]) + (x[-1] - x[-2])*(B[-2] + 2.0*B[-1])/6.0     

    return B
# End Function computeB_spline1D


# -------------
# Compute Function Spline interpolation f_spline1D, from B, (x,f), at xeval
# -------------
# Call: feval = compute_f_spline1D(splinePar, xeval)
def compute_f_spline1D(splinePar, xeval):
    """
    Input: 
    -> xeval        : (n,) array containing x values in which compute solution
    -> splinePar    : Dictionnary containing
           -> B         : spline parameter solution vector
           -> x,f       : (N,) array, training points
    Output:
    -> feval        : (n,) array - fSpline(xeval)
   """
    # Initialization
    B            = splinePar["B"]
    x            = splinePar["x"]
    f            = splinePar["f"]
    N            = len(f)
    n            = len(xeval)
    alphaNum     = (f[1] - f[0])/(x[1] - x[0]) + (x[1] - x[0])*(-B[1] - 2.0*B[0])/6.0     # Numerical f'(left)
    feval        = np.zeros(n)

    # Intervall in which find xeval (np.array form) (careful at left!!)
    ind          = x.searchsorted(xeval) 
    i0           = np.where(ind == 0)
    ix           = np.where(ind != 0)
    ind          = np.delete(ind,i0)

    # Left Case
    feval[i0]    = f[i0]

    # Computation init
    ti           = x[ind-1]
    ti1          = x[ind]
    Ai           = f[ind-1]
    Ai1          = f[ind]
    Bi           = B[ind-1]
    Bi1          = B[ind]
    t            = xeval[ix]
    hi           = ti1 - ti
    xi           = (t-ti)/hi

    # feval
    feval[ix]    = (1-xi)*Ai + xi*Ai1 + hi**2/6.0*((1-xi)**3 - (1-xi))*Bi + hi**2*(xi**3 - xi)*Bi1/6.0

    return feval
# End Function compute_f_spline1D


# -------------
# Compute Derivative - Spline interpolation df_spline1D, from B, (x,f), at xeval
# -------------
# Call: dfeval = compute_df_spline1D(splinePar, xeval)
def compute_df_spline1D(splinePar, xeval):
    """
    Input: 
    -> xeval        : (n,) array containing x values in which compute solution
    -> splinePar    : Dictionnary containing
           -> B         : spline parameter solution vector
           -> x,f       : (N,) array, training points
    Output:
    -> dfeval        : (n,) array - f'Spline(xeval)
   """
    # Initialization
    B            = splinePar["B"]
    x            = splinePar["x"]
    f            = splinePar["f"]
    N            = len(f)
    n            = len(xeval)
    alphaNum     = (f[1] - f[0])/(x[1] - x[0]) + (x[1] - x[0])*(-B[1] - 2.0*B[0])/6.0     # Numerical f'(left)
    dfeval       = np.zeros(n)

    # Intervall in which find xeval (np.array form) (careful at left!!)
    ind          = x.searchsorted(xeval) 
    i0           = np.where(ind == 0)
    ix           = np.where(ind != 0)
    ind          = np.delete(ind,i0)

    # Left Case
    dfeval[i0]   = alphaNum

    # Computation init
    ti           = x[ind-1]
    ti1          = x[ind]
    Ai           = f[ind-1]
    Ai1          = f[ind]
    Bi           = B[ind-1]
    Bi1          = B[ind]
    t            = xeval[ix]
    hi           = ti1 - ti
    xi           = (t-ti)/hi

    # dfeval
    dfeval[ix]   = (Ai1 - Ai)/hi + hi*(1 - 3.0*(1-xi)**2)/6.0*Bi + hi*(3.0*xi**2 - 1)*Bi1/6.0

    return dfeval
# End Function compute_df_spline1D



# -------------
# Compute Matrix A - no BC
# -------------
# Call: A = computeMatrixA_spline1D(x,f)
def computeMatrixA_spline1D(x,f):
    """
    Input: 
    -> x,f          : (N,) arrays such that f[i] = f(x[i])
    Output:
    -> A            : ((N-2), 4(N-2)) full matrix
   """
    # Initialization
    N              = len(x)
    A              = np.zeros((N-2)**2).reshape(N-2, N-2)
    h              = np.diff(x)       # hi = xi+1 - xi

    # Interior
    for i in np.arange(2,N-2):
        iloc           = i-1
        jloc           = np.array([iloc-1, iloc, iloc+1])
        A[iloc, jloc]  = np.array([h[i-1], 2.0*h[i-1] + 2.0*h[i], h[i]])/6.0
    # Left 
    i              = 1
    iloc           = i-1
    jloc           = np.array([iloc, iloc+1])
    A[iloc, jloc]  = np.array([2.0*h[i-1] + 2.0*h[i], h[i]])/6.0

    # Right
    i              = N-2
    iloc           = i-1
    jloc           = np.array([iloc-1, iloc])
    A[iloc, jloc]  = np.array([h[i-1], 2.0*h[i-1] + 2.0*h[i]])/6.0
    
    return A
# End Function computeMatrixA_spline1D


# -------------
# Compute Second Member b - no BC
# -------------
# Call: b = computeSnd_b_spline1D(x,f)
def computeSnd_b_spline1D(x, f):
    """
    Input: 
    -> x,f          : (N,) arrays such that f[i] = f(x[i])
    Output:
    -> b            : (N-2,) array
   """
    # Initialization
    N            = len(f)
    b            = np.zeros(N-2)
    h            = np.diff(x)

    # Loop
    for i in np.arange(1,N-1):
        i_loc      = i-1
        b[i_loc]   = (f[i+1] - f[i])/h[i] - (f[i] - f[i-1])/h[i-1]
    return b
# End Function computeSnd_b_spline1D


def interpSplines1D(xx,yy,alpha=0,beta=0,periodic=False):
    """
    -- Cubic Spline Interpolation 1D
    f,df=interpSplines1D(x,y,alpha=df0,beta=df1)
    yy = f(xx) ; dyy = df(xx) for evaluation

    -- Input: 
    xx,yy      : (n,) array of data
    alpha      : f'(xx[0] ) to fix
    beta       : f'(xx[-1]) to fix
    -- Output
    f, df      : (n,) -> (n,) functions for evaluation, and derivative
    """
    # Pre-Process
    B = computeB_spline1D(xx, yy, alpha, beta, periodic)
    splinePar     = {"x":xx, "f":yy, "B":B}
    
    # Function f
    def f_sp(x):
        f       = compute_f_spline1D(splinePar, x.reshape(len(x)))
        return f.reshape(x.shape)

    # Function df = f'
    def df_sp(x):
        df      = compute_df_spline1D(splinePar, x.reshape(len(x)))
        return df.reshape(x.shape)
    return f_sp, df_sp


def fmin_cobyla(restart=1,returnAll=False,**opt):
    """
    -- Uses several runs of fmin_cobyla_one
    restart      : if >1 and xmin,xmax given: x0 used for First run. Then restart-1 runs are performed from random. (nmax per run)
    returnAll    : If True, 
out, gCons = ...(returnAll=True), with gCons containing all functions constraints that should be positive
    """
    # Basic
    opt['returnAll'] = returnAll
    res = fmin_cobyla_one(**opt)
    if restart == 1: return res
    
    # Several restart
    n = restart-1
    try:
        xmin=opt['xmin']
        xmax=opt['xmax']
        x0  =sampleLHS(n,xmin,xmax)
    except: return res

    # Restart
    r = [res]
    for i in range(n):
        opt['x0'] = x0[i,:]
        r.append(fmin_cobyla_one(**opt))

    # ReturnAll
    if returnAll:
        g = [r[k][1] for k in range(n+1)]
        r = [r[k][0] for k in range(n+1)]
        
    # Choose
    t = [r[k]['t'] for k in range(n+1)]
    nfev = [r[k]['nfev'] for k in range(n+1)]
    f = [r[k]['fun'] for k in range(n+1) if r[k]['success']]
    if len(f)==0:
        f = [r[k]['fun'] for k in range(n+1)]
    i = np.argmin(f)
    out = deepcopy(r[i])

    # Output
    out['restart'] = restart
    out['nfev_tab'] = nfev
    out['fun_tab'] = np.array(f)
    out['nfev'] = np.sum(nfev)
    out['t']    = np.sum(t)
    if returnAll: return out,g[i]
    return out

        
def fmin_cobyla_one(f=None, x0=None, xmin=None, xmax=None, gCons=[], tol=1e-6, nmax=1000,disp=False, show=False, returnAll=False):
    """
    -- Perform Cobyla Constraint minimization, starting from x0 (mandatory)
    Min f(x) 
       s.t. x € Prod_i [xmin(i),xmax(i)]
            gCons[k](x)<0
    
    Here, wrappers are used 

    -- Input: 
    f            : (n,d)->(n,) Objective function also accepting (d,) arrays
    x0           : (d,) array - Starting point
    xmin,xmax    : None or list of size d (possibly containing None)
e.g. xmin=None, xmax=[None,-1,8]
    gCons        : List of nc constraints functions gk(x)<0, structure similar to f
    tol          : see Cobyla 'rhoend' for scipy.optimize.fmin_cobyla
    nmax         : maximal number of evaluations
    disp         : If True, message Cobyla given
    show         : If True, nice Solution print
    returnAll    : If True, 
out, gCons = ...(returnAll=True), with gCons containing all functions constraints that should be positive
    -- Output: Dictionary of results
    res         containing x, x0, fun, nfev, success, t
    """
    # Initialization
    nmax = int(nmax)
    assert x0 is not None
    t0 = time.time()
    
    # New cons
    def funNeg(f):
        def ff(x):
            return -f(x)
        return ff
    cons = [funNeg(gCons[k]) for k in range(len(gCons))]
    
    # Function Generator for xmin/xmax (so >0)
    def funGen(xcons,i,t='min'):
        if t=='min':
            def f(x):
                return x[i]-xcons[i]
                # try: return x[i]-xcons[i]
                # except: return np.array(x)[i]-xcons[i]
            return f
        def f(x):
            return -x[i]+xcons[i]
        return f

    # xmin
    if xmin is not None:
        for i in range(len(xmin)):
            if xmin[i] is not None:
                cons.append(funGen(xmin,i))

    # xmax
    if xmax is not None:
        for i in range(len(xmax)):
            if xmax[i] is not None:
                cons.append(funGen(xmax,i,t=0))

    # Conversion
    nc   = len(cons)
    consFmin = [{'type':'ineq','fun':cons[k]} for k in range(nc)]

    # Evaluation
    import scipy
    res = scipy.optimize.minimize(f, x0=x0, method='COBYLA',constraints=consFmin, options={'maxiter':nmax,'disp':disp},tol=tol)

    # OUtput
    out = {'res':res,'x':res.x,'fun':res.fun,'nfev':res.nfev,'success':res.success,'method':'cobyla','x0':x0,'t':time.time()-t0,'restart':1}
    if show: print(out)

    if returnAll: return out, cons
    
    return out


def interpLineSearch(df0,f0,fi,fi1,ai,ai1):
    if np.isnan(ai) or np.isnan(ai1):
        assert 0 == 1
        
    if abs(ai-ai1)<1e-8: return min(ai,ai1)
    k = ai**2*ai1**2*(ai-ai1)
    A = np.eye(2)
    A[0,:] = [ai1**2,-ai**2]
    A[1,:] = [-ai1**3,ai**3]
    f = np.array([fi-f0-df0*ai,fi1-f0-df0*ai1])
    B = np.dot(A,f)/k
    a = B[0]; b = B[1]
    x = (-b+sqrt(b**2-3*a*df0))/(3*a)
    if np.isnan(x):
        print('ERROR gF.interpLineSearch')
        assert 0==1
    return x


def backTracking_lineSearch(fun, xk, fk, dfk, pk, fk1=None,c1=1e-4,rho=0.8,modeDebug=False,aTab=[],fTab=[],nloop=50):
    phi_0  = fk
    dphi_0 = (dfk*pk).sum()
    if modeDebug: print('dphi_0[BackTracking]=',(pk*dfk).sum())
    if dphi_0>=0 and modeDebug:
        print('WARNING lineSearch[BackTracking]: pk is not a descent direction')
        print('dphi_0=%s'%(dphi_0))

    def phi(alpha):
        """Evaluation: save datas"""
        fa = fun(xk+alpha*pk)
        fTab.append(fa)
        aTab.append(alpha)
        return fa
    def condition():
        """First Wolfe Condition: if True, stop"""
        if fTab[-1] <= phi_0+c1*aTab[-1]*dphi_0: return True
        return False

    # Initial Guess
    a0 = rho
    f0 = phi(a0)
    if condition(): return a0,f0,fTab,aTab,True

    for k in range(nloop):
        a0 = a0*rho
        f0 = phi(a0)
        if condition():
            if modeDebug:
                print('[BackTracking: alpha=%s,f0=%s,f(alpha)=%s'%(a0,phi_0,f0))
            return a0,f0,fTab,aTab,True

        
    if modeDebug:
        print('In BackTracking[linesearch], failure finding suitable alpha')
    return a0,f0,fTab,aTab,False


def lineSearch(fun, xk, fk, dfk, pk, fk1=None,c1=1e-4,nmax=50,eps1=1e-3,eps0=1e-3, modeDebug=False,dfk1=None,pk1=None,alpha1=None):
    """
    -- Line Search: Find suitable alpha minimizing inaccurately
    alpha -> phi(alpha)=f(xk+alpha*pk)

    ak,phi_k,fTab,aTab,success = lineSearch(fun, xk, fk, dfk, pk, fk1=None)

    -- Input
    f                : (d,) -> float
    xk               : (d,) 
    fk               : float - f(xk)
    dfk              : (d,) f'(xk)
    pk               : (d,) descent direction, s.t. fk.pk<0
    fk1              : float - f(xk-1), optional
    c1               : for first Wolfe condition: phi(alpha)<=phi(0)+alpha.c1.phi'(0)
    nmax             : Maximal Number of evaluation (for first loop)
    eps1,eps0        : |alpha_i|>eps0 and |alpha_i-alpha_i-1|>eps1. Otherwise, alpha_i=alpha_i1/2
    dfk1             : Grad f(xk-1) [Optional]
    pk1              : pk-1
    alpha1           : alpha-1

    -- Output
    alpha            : alpha solution
    phi_k            : f(xk+alpha*pk)
    fTab             : (N,) array.  History of f(xk+alpha*pk)
    aTab             : (N,) array.  History of alpha
    success          : True=success, False=failure
    """
    # Initialization
    fTab=[]; aTab=[]
    phi_0  = fk
    dphi_0 = (dfk*pk).sum()
    if modeDebug: print('dphi_0=',(pk*dfk).sum())
    if dphi_0>=0 and modeDebug:
        print('WARNING lineSearch: pk is not a descent direction')
        print('dphi_0=%s'%(dphi_0))

    def phi(alpha):
        """Evaluation: save datas"""
        fa = fun(xk+alpha*pk)
        fTab.append(fa)
        aTab.append(alpha)
        return fa
    def condition():
        """First Wolfe Condition: if True, stop"""
        if fTab[-1] <= phi_0+c1*aTab[-1]*dphi_0: return True
        return False
    def regul(ai,ai1):
        """Regularization: if too close alpha_i, alpha_i+1, or alpha_i to 0"""
        if abs(ai-ai1)<eps1 or abs(ai)<eps0:
        # if abs(ai-ai1)<eps1:
            ai=0.5*ai1
        return ai

    # Initial Guess
    a0 = 1
    if fk1 is not None:
        a0 = 2*(fk-fk1)/dphi_0
    # if dfk1 is not None and pk1 is not None and alpha1 is not None:
    #     a0 = alpha1*(dfk1*pk1).sum()/dphi_0
    a0 = min(1,1.01*a0)
    if a0 <0 and modeDebug:
        # a0=1
        print("guess<0")
    f0 = phi(a0)
    if condition(): return a0,f0,fTab,aTab,True

    # Quadratic Interpolation
    a1 = 0.5*(dphi_0*a0**2)/(f0-phi_0-dphi_0*a0)
    a1 = regul(a1,a0)
    f1 = phi(a1)
    if condition(): return a1,f1,fTab,aTab,True

    # Loop Cubic Interpolation
    ai = a1 ; ai1 = a0 ; fi1 = f0 ; fi = f1
    for i in range(nmax):
        try: aa = interpLineSearch(dphi_0,phi_0,fi,fi1,ai,ai1)
        except: aa = 0.5*(dphi_0*ai**2)/(fi-phi_0-dphi_0*ai)
        aNew = regul(aa,ai)
        ai1 = ai 
        fi1 = fi  
        ai = aNew
        fi = phi(ai)
        if condition(): return ai,fi,fTab,aTab,True
        if len(fTab)>nmax: break

    # No first Wolfe Conditions
    k  = np.argmin(np.array(fTab))

    k  = np.array([i for i in range(len(aTab)) if abs(aTab[i])>1e-8])
    if len(k)>0:
        k = k.argmin()
        ak = aTab[k] ; phi_k = fTab[k]
        if phi_k < phi_0:
            return ak,phi_k,fTab,aTab,True

    # Seek for minimizer close to 0
    ak = 1
    for k in range(10):
        ak = ak*0.9
        fk = phi(ak)
        if condition(): return ak,fk,fTab,aTab,True

    # Failure
    return 0,phi_0,fTab,aTab,False
    
                    
class BFGS:
    """
    -- BFGS Algorithm: Algo 6.1 [Book Nocedal, 2006]
    Minimize f(x) starting from x0. Stops if ||df(xk)||<eps or n_eval_f>NMAX (even if gradient evaluations is slower).
    Line search is performed Interpolation Method p.55-57 [First Wolfe Condition]
    Initial Hessian inverse matrix is automatically initialized

    -- Input
    x0                 : (d,) array - Initial Guess
    f                  : (d,) -> (1,) or float: Function
    df                 : (d,) -> (d,) : Gradient 
    c1,c2              : For Wolfe Conditions
    NMAX               : Maximal number of function evaluations
    eps                : Stopping criterion on the gradient

    -- Output
    xk                 : 
    """
    def __init__(self,x0=None, NMAX=100, eps=1e-5, f=None, df=None, c1=1e-4, c2=0.9, h=1e-4, H0=None, f0=None,modeDebug=False):
        """
        h                   : For Numerical Gradient
        """
        # Initialization
        self.history = {'x':[x0],'f_it':[],'f':[],'df':[],'df_it_norm':[],'df_it':[],'alpha':[],'n_alpha':[],'n_dalpha':[]}
        self.n = 0
        self.nn = 0
        self.d = len(x0)
        self.NMAX = NMAX
        self.eps = eps
        self.fun = f
        self.c1 = c1
        self.c2 = c2
        self.x0 = x0
        if f0 is None: f0 = self.f(x0)
        self.fk_1 = f0
        self.fk_2 = None
        self.history['f_it'].append(self.fk_1)
        self.H0 = H0
        self.modeDebug=modeDebug
        
        # GRadient
        if df is None:
            from newtonSolver import gradNumFunction
            dff = gradNumFunction(self.f, h = h, d =self.d, p =1)
            def df(x):
                y = dff(x)
                return y.reshape(-1)
        self.dfun = df

        # Datas for guess alpha0
        self.pk1 = None
        self.alpha1=None
        self.dfk1 = None
        
        # History
        return

    def f(self,x):
        self.n +=1
        y = self.fun(x)
        self.history['f'].append(y)
        return y

    def df(self,x):
        self.nn +=1
        y = self.dfun(x)
        self.history['df'].append(np.linalg.norm(y))
        return y

    def wolfe(self,xk,alpha_k,pk,dfk=None,fk=None,c1=None,c2=None):
        """Check for Strong Wolfe Conditions: Requires Gradient Estimation and 2 Function Evaluations
        Output: 
        Wolfe, Strong Wolfe [Booleans]
        """
        if c1 is None: c1 = self.c1
        if c2 is None: c2 = self.c2
        if fk is None: fk = self.f(xk)
        if dfk is None: dfk = self.df(xk)
        a = self.f(xk+alpha_k*pk)
        b = (self.df(xk+alpha_k*pk)*pk).sum()
        val1 = a<= fk + c1*alpha_k*(dfk*pk).sum()
        val2 = b>= c2*(dfk*pk).sum()
        val3 = abs(b) <= abs(c2*(dfk*pk).sum())

        if val1 and val2 and val3: return True,True
        if val1 and val2: return True, False
        return False, False
        
    def searchLine(self,xk,pk,dfk,alphaMax=10,nIter=None):


        alpha,fxk1,fTab,aTab,success = lineSearch(self.f, xk, self.fk_1, dfk, pk, fk1=self.fk_2, modeDebug=self.modeDebug,alpha1=self.alpha1,dfk1 = self.dfk1, pk1=self.pk1)

        if abs(alpha)<1e-8: success=False
        # if len(fTab)>6 or np.array(aTab).min()<0:
        if len(fTab)>6 and self.modeDebug:
            print('Iteration %s'%(nIter))
            print('fTab=',fTab)
            print('aTab=',aTab)
            print('dphi_0=',(pk*dfk).sum())
            plt.scatter(aTab,fTab,label='Iteration',marker='o',s=10)
            plt.title('n=%s, alpha=%2.3E, fk=%2.3E,fk+1=%2.3E'%(len(aTab),alpha,self.fk_1,fxk1))
            plt.show()
            

        # If Succcessful
        if success:
            self.alpha1 = alpha
            self.dfk1 = dfk
            self.pk1 = pk
            self.fk_2 = self.fk_1
            self.fk_1 = fxk1
            self.history['alpha'].append(alpha)
            self.history['n_alpha'].append(len(fTab))
            self.history['f_it'].append(fxk1)
            return alpha

        # BackTracking
        alpha,fxk1,fTab,aTab,success = backTracking_lineSearch(self.f, xk, self.fk_1, dfk, pk, fk1=self.fk_2, modeDebug=self.modeDebug,rho=0.9,aTab=aTab,fTab=fTab)
        self.alpha1 = alpha
        self.dfk1 = dfk
        self.pk1 = pk
        self.fk_2 = self.fk_1
        self.fk_1 = fxk1
        self.history['alpha'].append(alpha)
        self.history['n_alpha'].append(len(fTab))
        self.history['f_it'].append(fxk1)

        # if len(fTab)>6 or np.array(aTab).min()<0:
        if len(fTab)>6 and self.modeDebug:
            print('BACKTRACKING')
            print('Iteration %s'%(nIter))
            print('fTab=',fTab)
            print('aTab=',aTab)
            print('dphi_0=',(pk*dfk).sum())
            plt.scatter(aTab,fTab,label='Iteration',marker='o',s=10)
            plt.title('n=%s, alpha=%2.3E, fk=%2.3E,fk+1=%2.3E'%(len(aTab),alpha,self.fk_1,fxk1))
            plt.show()

        if success: return alpha

        if self.modeDebug:
            print('Line Search Failure: Have a look in self.[_fTab,_aTab,_f0,_phi,_pk,_dfk]')

        self._fTab=fTab
        self._aTab = aTab
        self._f0  = self.fk_2
        def phi(alpha):
            """Evaluation: save datas"""
            try: fa = self.f(xk+alpha*pk)
            except:
                 fa = [self.f(xk+alpha[k]*pk) for k in range(len(alpha))]
            return fa
        self._phi = phi
        self._pk = pk
        self._dfk = dfk

        if self.modeDebug: 
            print('Failure searchLine[BFGS]: Have a look in self._fTab,self._aTab, self._f0, self.phi')
        
        return 0
        
        # import scipy
        # alpha,fc,gc,fxk1,fxk,_ = scipy.optimize.line_search(self.f, self.df, xk, pk, gfk=dfk, old_fval=self.fk_1, old_old_fval=self.fk_2, args=(), c1=0.0001, c2=0.9, amax=50)
        # self.fk_2 = fxk
        # self.fk_1 = fxk1
        # self.history['alpha'].append(alpha)
        # self.history['n_alpha'].append(fc)
        # self.history['n_dalpha'].append(gc)
        # self.history['f_it'].append(fxk1)
        
        # return alpha
        

    def __call__(self):

        t0 = time.time()
        xk = self.x0.copy()
        if self.H0 is None: H  = np.eye(self.d)
        else: H = self.H0.copy()
        dfk = self.df(xk)
        success = True

        for k in range(self.NMAX):
            if self.n > self.NMAX: break
            if np.linalg.norm(dfk)<self.eps: break

            # Search Direction
            pk = -np.dot(H,dfk)

            if (pk*dfk).sum()>0 and self.modeDebug:
                print('iteration %s: pk.gradk>0: check eigenvalues H'%(k))
                print(np.linalg.eigvals(H))
            
            # Line search
            alpha_k = self.searchLine(xk,pk,dfk,nIter=k)

            if alpha_k == 0:
                success=False
            
            # Mini Update
            xk1 = xk + alpha_k*pk
            dfk1 = self.df(xk1) ;
            self.history['df_it'].append(dfk1)
            self.history['df_it_norm'].append(np.linalg.norm(dfk1))
            sk  = alpha_k*pk
            yk  = dfk1-dfk
            rhok=1.0/(yk*sk).sum()

            if (yk*sk).sum()==0 and self.modeDebug:
                print('(yk*sk).sum()=0: ')
                print('yk',yk)
                print('sk',sk)
                print('alpha_k',alpha_k)
                print('pk',pk)
                print('iteration ',k)
                
            # Inverse Hessian Update
            if k==0 and self.H0 is None: H = (yk*sk).sum()/(yk*yk).sum()*np.eye(self.d)
            A = np.eye(self.d) - rhok*np.dot(sk.reshape(-1,1),yk.reshape(1,-1))
            B = np.eye(self.d) - rhok*np.dot(yk.reshape(-1,1),sk.reshape(1,-1))
            H = np.dot(np.dot(A,H),B) + rhok*np.dot(sk.reshape(-1,1),sk.reshape(1,-1))

            # Update
            xk = xk1
            dfk = dfk1

            # Termination
            # code.interact(local=dict(globals(), **locals()))
            if self.n > self.NMAX: break
            if np.linalg.norm(dfk)<self.eps: break

        self.H = H
        self.nIter=k+1

        # Output:
        res = {'x': xk, 'fun': self.fk_1, 'nfev': self.n, 'nGrad': self.nn, 'method': 'myBFGS', 't': time.time()-t0, 'nIter': k+1, "norm_jac": np.linalg.norm(dfk),"jac": dfk}
        x = xk ; 
        dfx = self.dfun(x)
        print('||x-xsol||=',np.linalg.norm(xk-x))
        print('||grad(x)-dfx||=',np.linalg.norm(dfx-dfk))
        print('||grad(x)||=',np.linalg.norm(dfx))
        print('||dfk||=',np.linalg.norm(dfk))
        
        # Failure
        if self.history['f_it'][0]<res['fun']:
            res['ERROR']  = 'f(x0)=%s<f(xSol)=%s'%(self.history['f_it'][0],res['fun'])
        # Error Line Search
        if success==False :
            if self.modeDebug: print('In Call BFGS: Failure linesearch. Stop')
            res['alphaTab'] = self._aTab
            res['phi_alphaTab'] = self._fTab

        self.res = deepcopy(res)
        return res

    @property
    def xSol(self):
        """Solution"""
        try: return self.res['x']
        except:
            return None

    @property
    def xSol_grad(self):
        """Derivative xSol"""
        try: return self.res['jac']
        except:
            return None
        
    
    def __str__(self):
        m = '  %s: [d=%s] -- eps=%2.3E, NMAX=%s\n'%(self.__class__.__name__,self.d,self.eps, self.NMAX)
        try:            a = self.res
        except: return m

        m+= "f(x)=%2.3E [nIter=%s,nf=%s,ng=%s] - |f'(x)|=%2.3E [%s]  x=\n"%(self.res['fun'],self.res['nIter'],self.n,self.nn, self.res["norm_jac"],convertTime(self.res['t']))
        m+= printTab1D(self.res['x'])
        return m

    def print(self):
        return 
    
    def __repr__(self):
        return self.__str__()

    def nIterTab(self):
        return np.arange(self.nIter)

    def plot(self, log=True):

        # ----------
        # Convergence
        # ----------
        # Initialization
        plt.figure(1)
        title = r'Convergence'
        plt.suptitle(title)
        
        # f vs nIt
        nIt = np.arange(self.nIter+1)
        plt.subplot(221)
        plt.scatter(nIt,self.history['f_it'],marker='x',color='black',s=5)
        plt.xlabel(r'$N_{iter}$')
        plt.ylabel(r'$f(x_i)$')

        plt.subplot(222)
        plt.semilogy(nIt,self.history['f_it'],marker='x',color='black',lw=3)
        plt.xlabel(r'$N_{iter}$')
        # plt.ylabel(r'$f(x_i)-log$')

        # |f'| vs nIt
        plt.subplot(223)
        plt.scatter(nIt[1:],self.history['df_it_norm'],marker='x',color='black',s=5)
        plt.xlabel(r'$N_{iter}$')
        plt.ylabel(r"$|f'(x_i)$")

        plt.subplot(224)
        plt.semilogy(nIt[1:],self.history['df_it_norm'],marker='x',color='black',lw=3)
        plt.xlabel(r'$N_{iter}$')
        # plt.ylabel(r'$f(x_i)-log$')
        
        # ----------
        # Line Search
        # ----------
        # Initialization
        plt.figure(2)
        title = r'Line Search'
        plt.suptitle(title)
        
        # f vs nIt
        nIt = np.arange(self.nIter)
        plt.subplot(211)
        plt.scatter(nIt,self.history['alpha'],marker='x',color='black',s=5)
        plt.xlabel(r'$N_{iter}$')
        plt.ylabel(r'$\alpha$')

        plt.subplot(212)
        plt.scatter(nIt,self.history['n_alpha'],marker='x',color='black',s=5)
        plt.xlabel(r'$N_{iter}$')
        plt.ylabel(r'$N_{eval}$')

        plt.show()

# END--------------OPTIONS----------------------


# -----------------PLOT: SubFunctions/Variables-
# -------------
# Global Variables
# -------------
boolScript          = False
nameScript          = None
folderScript        = None
indScript           = None

# -------------
# Functions associated to set
# -------------
def setBoolScript(boolVal):
    global boolScript
    boolScript          = boolVal 
    return
def setNameScript(name):
    global nameScript
    nameScript          = name
    return
def setFolderScript(folder):
    global folderScript
    folderScript        = folder
    return
def setIndScript(ind):
    global indScript
    indScript           = ind 
    return

# -------------
# Functions associated to get
# -------------
def getBoolScript():
    return boolScript
def getNameScript():
    return nameScript
def getFolderScript():
    return folderScript
def getIndScript():
    return indScript  

# -------------
# get name file script
# -------------
def getFileScript():
    name           = getNameScript()
    folder         = getFolderScript()
    fileScript     = folder + name + ".py"
    return fileScript


# -------------
# get Image Out, when save image in the script
# -------------
def getImageOutScript():
    name           = getNameScript()
    folder         = getFolderScript()
    imageOut       = folder + name + "_image.eps"
    return imageOut

# -------------
# Create the file Script
# -------------
def createFileScript():
    fileScript     = getFileScript()
    command        = "rm -f " + fileScript
    os.system(command)
    command        = "touch " + fileScript
    os.system(command)
    return

# -------------
# Open File script, and get all data
# -------------
def getDataFileScript():
    fileScript     = getFileScript()
    with open(fileScript, 'r') as fin:
        data = fin.read().splitlines(True)
    return data

# -------------
# Write all datas in file script
# -------------
def writeFileScript(data):
    fileScript     = getFileScript()
    with open(fileScript, 'w') as fout:
        fout.writelines(data) 
    return

# -------------
# Save points in folder/name_ind.dat
# -------------
def savePointsDataScript(points):
    name           = getNameScript()
    folder         = getFolderScript()
    ind            = getIndScript()
    fileSave       = folder + name + "_points" + str(ind) + ".dat"
    np.savetxt(fileSave, points, delimiter = " ")
    return fileSave
# END: ----------- PLOT: SubFunctions/Variables-


# -----------------PLOT-------------------------
# -------------
# Init plot
# -------------
# Call: initPlot(labelAxis = True, script = True, name = "test1", folder = "../dat/"}
def initPlot(labelAxis = False, equalAxis = False, figsize = (8,6), grid = True,
             label_x = [r'$\mathbf{x_1}$',20], label_y = [r'$\mathbf{x_2}$',20], xTicks=True, yTicks=True, frameon=True, tickSize=10, left=0.15, bottom=0.15, right=0.95, top=0.92):
    """
    initPlot(labelAxis = False, equalAxis = False, figsize = (10,10), grid = True,             
label_x = [r'$\mathbf{x_1}$',20], label_y = [r'$\mathbf{x_2}$',20])


    Input: 
    -> **opt           : Dictionary
         labelAxis          : Boolean - if True, bold axis, with (x1,x2)
         name               : string - name for script file basis: 
         folder             : string - "./", "/home/nrazaaly/test/"
         script             : Boolean - if True, a python script to plot is created in folder, with associated files.dat 
    """
    # Plot
    plt.rc('font', weight='bold')
    plt.rc('legend', fontsize=tickSize)
    plt.rc('axes', labelsize=tickSize)
    fig, ax    = plt.subplots()
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    if equalAxis: 
        plt.axis('equal')
        #plt.gca().set_aspect('equal', adjustable='box') 
    plt.rc('font', weight='bold')
    plt.figure(1, figsize = figsize)
    plt.cla()
    plt.grid(grid)
    
    # Frame
    if frameon == False:
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
    
    # Ticks: 
    if xTicks == False: plt.xticks([])
    if yTicks == False: plt.yticks([])
    
    # Label Axis
    if labelAxis:
        plt.xlabel(label_x[0], fontweight='bold', fontsize = label_x[1])
        plt.ylabel(label_y[0], fontweight='bold', fontsize = label_y[1])

    return
# End function initPlot

# Simple Plot
def prePlot(left=0.15,bottom=0.16,top=0.9,right=0.99,scientific_x=False,scientific_y=True,grid=False, bold=True, powerlimit=(-3,2), tickSize=20, equalAxis=False, frameon=True, xTicks=True, yTicks=True):
    """
    powerlimit             : for scientific writing
    scientific_x or _y for axis scientific mode
    frameon                : Axes or not
    xTicks                 : False -> Erase x-ticks
    yTicks                 : False -> Erase y-ticks
    """
    # distance between x and y axis and the numbers on the axes
    matplotlib.rcParams['xtick.major.pad'] = 5
    matplotlib.rcParams['ytick.major.pad'] = 5

    # Init Plot
    fig,ax = plt.subplots()
    fig.subplots_adjust(left=left,bottom=bottom,top=top,right=right)    
    plt.grid(grid)
    if equalAxis: plt.axis('equal')
    if bold: plt.rc('font', weight='bold')
    plt.rc('legend', fontsize=tickSize)
    plt.rc('axes', labelsize=tickSize)
    ax.tick_params(axis='both', which='major', labelsize=tickSize)  
    ax.tick_params(axis='both', which='minor', labelsize=tickSize)  

    # Scientific Writing
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits(powerlimit) 
    if scientific_y: ax.yaxis.set_major_formatter(formatter)
    if scientific_x: ax.xaxis.set_major_formatter(formatter)

    # Ticks: 
    if xTicks == False: plt.xticks([])
    if yTicks == False: plt.yticks([])
    
    # Frame
    if frameon == False:
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
    
    return 



def plotBlade(xPS=None, yPS=None, xSS=None, ySS=None, show=True, color='black', label='', save=False, fileSave='blade.eps', dpi=300, pitch=0.045, frameon=False, xTicks=False, yTicks=False, degree=-90, kPitch=[1], zorder=0, lw=2, ticksize=15):
    """
    Plot Blade with Rotation degree
    
    Input: 
    lw                          : LineWidth
    {x,y}{PS,SS}                : (n,) arrays, Pressure/Suction Sides
    pitch                       : Values for pitch
    degree                      : Rotation Angle (degree)
    frameon                     : Bool - Rectangular Frame
    x,yTicks                    : Bool
    kPitch                      : array for pitch to plot - kPitch=[-1,1] for 3 blades
    """
    # Initialization
    from matplotlib import  transforms
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(-90)
    plt.axis('equal')
    if frameon == False:
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
    if xTicks == False: plt.xticks([])
    if yTicks == False: plt.yticks([])
    opt = {'color': color, 'transform': rot+base, 'zorder':zorder, 'linewidth': lw}
    plt.rc('xtick', labelsize=ticksize)
    plt.rc('ytick', labelsize=ticksize)
    plt.rc('axes', labelsize=ticksize)
    plt.rc('font', weight='bold')
    
    # Basic
    plt.plot(xPS,yPS, label=label, **opt)
    plt.plot(xSS,ySS,**opt)

    for k in kPitch:
        plt.plot(xPS,yPS+k*pitch,**opt)
        plt.plot(xSS,ySS+k*pitch,**opt)

    # Save
    if save:
        plt.legend(frameon=False)
        plt.savefig(fileSave,dpi=dpi)
        plt.close()
    if show:
        plt.legend(frameon=False)
        plt.show()
    return




# -------------
# Function add (value,label) to existing ticks
# -------------
def addTicks(axes,value,label,pos='x', fun=None):
    """
    Ex: 
    addTicks(ax,[5],[r'$N_i$'],pos='y',fun=int)
    addTicks(ax,[5,10],[r'$N_i$',10.1],pos='x')
    -> axes       : ax object
    -> value      : list of float
    -> label      : list of corresponding labels
    -> pos        : 'x' or 'y', according to axis you want to modify
    -> fun        : None [Default] - Can be (int) if you want to get only int values
    """
    # Extract
    if pos == 'x': 
        a=axes.get_xticks().tolist()[1:-1]
    else:
        a=axes.get_yticks().tolist()[1:-1]

    # fun
    if fun is not None:
        a = [fun(a[i]) for i in range(len(a))]

    # Set
    if pos == 'x':
        axes.set_xticks(a+value)
        axes.set_xticklabels(a+label)
    else:
        axes.set_yticks(a+value)
        axes.set_yticklabels(a+label)
    return 
# End function addTicks


# -------------
# Contour Plot 2D
# -------------
def contour2D_Plot(fun=None, xmin=None, xmax=None, Npts=80, cmap='jet', N_level=20, vmin=None, colorMin='white', zorder=0, colorbar=True, x=None, y=None, z=None, vmax=None, interpolation='bilinear', RETURN_ALL=False):
    """
    Input: 
    -> xmin*           : (2,) array: x_begin, y_begin
    -> xmax*           : (2,) array: x_end, y_end
    -> fun*            : Function: (N,2) -> float
    -> Npts*           : Number of points in each direction: Npts**2 total evaluations
    -> cmap            : 'gist_ncar_r', or other, kind of color bar
    -> N_levels        : Number of levels
    -> vmin, colorMin  : minimal value of contour below which the selected color is colorMin [white Default]
    -> zorder          : integer - where to put plot
    -> colorBar        : Bool - True => colorbar()
    -> z               : fun(meshgrid(xmin,xmax)), reshaped...
    -> x, y            : idem [only used for min,max]
    -> interpolation   : 'nearest','bilinear','bicubic','gaussian','spline36'
    (*): Pointless if z not None
    """
    from pylab import cm
    
    # Computations
    if z is None:
        x                  = np.linspace(xmin[0], xmax[0], Npts)
        y                  = np.linspace(xmin[1], xmax[1], Npts)
        x,y                = np.meshgrid(x,y)
        x0                 = x.reshape(x.size,1)
        y0                 = y.reshape(y.size,1)
        u                  = np.hstack((x0,y0))
        z                  = fun(u)
        z                  = z.reshape(x.shape)

    # cmap, for opt
    cc  = cm.get_cmap(cmap, N_level)
    cc.set_under(color=colorMin)
    opt = {'cmap': cc, 'zorder': zorder}
    if vmin is None: vmin = z.min()
    if vmax is None: vmax = z.max()
    opt['vmin']           = vmin
    opt['vmax']           = vmax

    # Plot
    V                     = np.linspace(z.min(), z.max(), N_level)
    plt.imshow(z, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],interpolation=interpolation, aspect='auto', **opt)
    if colorbar: plt.colorbar()

    # Output
    if RETURN_ALL == False: return 
    return x,y,z
# End function contour2D_Plot
# -------------


# -------------
# Contour Plot 3D
# -------------
def contour3D_Plot(fun, xmin, xmax, Npts=50, cmap='jet', N_level=10, alpha=0.5, ax=None, zorder=None):
    """
    Input: 
    -> xmin            : (2,) array: x_begin, y_begin
    -> xmax            : (2,) array: x_end, y_end
    -> fun             : Function: (N,2) -> float
    -> **opt           : Dictionary
         Npts               : Number of points in each direction: Npts**2 total evaluations
         cmap               : 'gist_ncar_r', or other, kind of color bar
         N                  : Number of levels
    """
    # Initialization 3D contours
    from pylab import cm
    if ax is None: 
        fig                = plt.figure()
        ax                 = fig.gca(projection='3d')
        mem=True
    else: mem=False
    
    # Computations
    x                  = np.linspace(xmin[0], xmax[0], Npts)
    y                  = np.linspace(xmin[1], xmax[1], Npts)
    x,y                = np.meshgrid(x,y)
    x0                 = x.reshape(x.size,1)
    y0                 = y.reshape(y.size,1)
    u                  = np.hstack((x0,y0))
    z                  = fun(u)
    z                  = z.reshape(x.shape)

    # Plot the surface
    N                  = z/z.max()
    ax.plot_surface(x, y, z, alpha=alpha,facecolors=cm.jet(N),zorder=None)
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(z)
    if mem: plt.colorbar(m)
    
    # Output
    return ax
# End function contour3D_Plot


# -------------
# Scatter Plot
# -------------
def scatterPlot(points_0, label = "", c = "black", s = 1, m = '.', annotate = False, e = [0,0], fontsize = 10, onlyAnnotate=False,zorder=None, connect=False, lw=1,scatter=True):
    """
    scatterPlot(points_0, label = "", c = "black", s = 1, m = '.', annotate = False):
    Scatter Plot: c = color, s = size point, m = marker point

    Ex: 
    scatterPlot(blade, c = "black", s = 1, m = '.'):
    scatterPlot([x,y], c = "black", s = 1, m = '.', annotate = True):
    """
    # Extraction points_0
    if len(points_0) == 2: # list of (N,1) array
        n               = len(points_0[0])
        points          = np.zeros((n,2), float)
        points[:,0]     = points_0[0]
        points[:,1]     = points_0[1]
    else:                  # (N,2) array
        points          = points_0

    # zorder
    opt={}
    if zorder is not None:
        opt['zorder'] = zorder
        
    # Label Case
    if onlyAnnotate == False:
        if scatter: 
            if label != "":
                opt['label'] = label
            plt.scatter(points[:,0], points[:,1], color = c, s = s, marker = m, **opt)

    # If Connected
    if connect:
        n = points.shape[0]
        for k in range(n):
            if k != n-1:
                x_values = [points[k,0], points[k+1,0]]
                y_values = [points[k,1], points[k+1,1]]
            else:
                x_values = [points[-1,0], points[0,0]]
                y_values = [points[-1,1], points[0,1]]
            plt.plot(x_values, y_values, color = c, lw=lw)
        
    # If Annotation
    if annotate:
        for i in range(points.shape[0]):
            plt.annotate(str(i), (points[i,0]+e[0],points[i,1]+e[1]), fontsize=fontsize, zorder=100,weight='bold')
                
        
    return
# End function scatterPlot


# -------------
# Plot Plot
# -------------
# Call: plotPlot(points, c = "grey", lw = 2, m = "o", label = "Case 1")
def plotPlot(points_0, label="", c=None, s=1, m='.'):
    """
    Call: plotPlot(points, c = "grey", s= 2, m = "o", label = "Case 1")
    """
    # Initialization
    if c is None: c = next(colorCycle)

    # Extraction points_0
    if len(points_0) == 2: # list of (N,1) array
        n               = len(points_0[0])
        points          = np.zeros((n,2), float)
        points[:,0]     = points_0[0]
        points[:,1]     = points_0[1]
    else:                  # (N,2) array
        points          = points_0

    # Label Case
    if label == "":
        plt.plot(points[:,0], points[:,1], color = c, lw = s, marker = m)
    else:
        plt.plot(points[:,0], points[:,1], color = c, lw = s, marker = m, label = label)

    # Script to create, if boolScript = True
    if getBoolScript():
        # Get data (what is written), update indScript
        setIndScript(getIndScript()+1)
        ind              = getIndScript()
        data             = getDataFileScript()

        # Save points
        filePoints       = savePointsDataScript(points)

        # data: load points, Write plot
        data.append("# Load points_" + str(ind))
        data.append("\npoints_"+str(ind)+" = np.loadtxt('"+filePoints+"')\n\n")
        data.append("# Plot points_" + str(ind))
        data.append("\nplt.plot(points_"+str(ind)+"[:,0], points_"+str(ind)+"[:,1], color = '"+c+"', lw = "+str(s)+", marker = '"+m+"', label = '"+label+"')\n\n")

        # Write data in file
        writeFileScript(data)
        
    return
# End function plotPlot


# -------------
# End plot
# -------------
def endPlot(save=False, show=True, fileIm='fig.eps', dpi=300, title='', legendOut=True, legendRight=True, extension='eps'):
    """
    Legend: if out, either on right (reducing x-axis), or centered in bottom (reducing y-axis)
    Call endPlot(save = True, file = "file.eps", show = False, dpi = 600, legendOut = True, legendRight = False, title = "My Title", extension= "eps")
    """
    # Title
    plt.title(title)

    # Legend: if out, reduce 80% on the side, put on the right
    if legendOut:  # Legend out the box
        ax  = plt.subplot(111)
        box = ax.get_position()
        if legendRight: # Put on the right - reduce x-axis 80%
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:           # Put on the bottom, centered - reduce 90% y-axis
            ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    else:          # Legend Inside, upper left
        plt.legend(loc="upper left", borderaxespad=0.)

    # Save
    if save:
        plt.savefig(fileIm, format=extension, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close()

    # Close
    #plt.clf()
    #if len(plt.get_fignums()) != 0:
    #plt.close('all')
        
    # Script to create, if boolScript = True
    if getBoolScript():
        # Get data, imageOut (where to store figure)
        data             = getDataFileScript()
        imageOut         = getImageOutScript()
        
        # data: title, legendOut=True, legendRight=True
        data.append("# Title\nplt.title('"+title+"')\n\n")
        data.append("# -----\n# Legend \n# -----\n")
        data.append("# If LegendOut = True: \nax  = plt.subplot(111)\nbox = ax.get_position()\n\n")
        data.append("# If LegendRight = True:\nax.set_position([box.x0, box.y0, box.width * 0.8, box.height])\nax.legend(loc='center left', bbox_to_anchor=(1, 0.5))\n\n")
        data.append("# If LegendRight = False (in comment): \n'''\nax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])\nax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)\n'''\n\n")
        data.append("# If LegendOut = False (in comment): \n'''\nplt.legend(loc='upper left', borderaxespad=0.)\n'''\n\n")

        # data: Save, Show, End
        data.append("# Save = True:\nplt.savefig('"+imageOut+"', format='eps', dpi="+str(dpi)+")\n\n")
        data.append("# Show = True:\nplt.show()\n\n")
        data.append("# End\nplt.clf()\n\n")

        # Write data in file
        writeFileScript(data)

    return
# End function endPlot
# -------------


# -------------
# Plot Histogram, and KDE (Gaussian - scipy)
# -------------
# plotHisto(x, N_bins = 10, alpha_bins = 0.3, N_x = 1000, show = True, KDE = True)
def plotHisto(x, N_bins=10, alpha_bins=0.3, N_x=1000, show=True, kde_plot=True):
    """
    Input:
     -> x           : (n,) array
    Output
     -> plot KDE, histogram of data x
    """
    # KDE Init
    if kde_plot:
        x_min           = x.min()
        x_max           = x.max()
        xx              = np.linspace(x_min, x_max, N_x)
        kde             = ss.gaussian_kde(x)

    # Plot
    t, ax           = plt.subplots(figsize=(8,6))
    if kde_plot:
        ax.plot(xx, kde(xx), label = 'KDE - scipyStats')
    ax.hist(x, normed=True, bins=N_bins, alpha=alpha, label = 'Histo')
    ax.legend(loc = 'best')

    # Show
    if show:
        plt.show()
    
    return
# End function plotHisto


# -------------
# Color Scatter Plot 
# -------------
def colorScatterPlot(x,z,cmap = 'jet', N_level = 20):
    plt.scatter(x[:,0], x[:,1],c = z, cmap = cm.get_cmap(cmap, N_level))
    plt.colorbar()
    return
# End colorScatterPlot


# Function endPlot
def ePlot(fileSave, save = False, show = True, extension='eps', dpi=1200, fontsize=15, markerSize=None, loc='best', legend=True):
    """
    Ex: ePlot(fileSave, save=save, show=show, extension=extension, dpi=dpi, fontsize=fontsize, markerSize=markerSize)
    function endPlot: Legend [Control Marker size, fontsize]
    -> save/show
    fontsize   -> legend
    markerSize -> legend lines/scatters
    """
    if legend is None: legend=False
    if legend:
        lgnd = plt.legend(loc=loc, scatterpoints=1, fontsize=fontsize)
    if markerSize is not None: 
        for handle in lgnd.legendHandles:
            handle.set_sizes([markerSize])
    if save: plt.savefig(fileSave, format=extension, dpi=dpi)
    if show: plt.show()
    return
# End Function ePlot


def plotLSS(G=None, mpfp=None, level=None, axes=None, 
            nCol=None, nRaw=None, couples=None,colorSafe='lightgreen', colorFail='lightsalmon',
            fontsize=15,figsize=15, a=-5, b=5, Npts=100, show=True, save=False, fileSave='fig.png', dpi=300, xTicks=None, yTicks=None,
            optG = {'colors': 'black', 'linewidths': 5, 'linestyles': '-', 'zorder': 1}, centered=False,c=2,
            listG=[], listOpt=[], listLevel=[], optDefault={'colors': 'black', 'linewidths': 3, 'linestyles': '--', 'zorder': 1} ):
    """
    Multi-Dimensional LSS plot
    
    -- Tutorial
    axes = plotLSS(G=G,mpfp=mpfp,level=level,show=False)
    axes = plotLSS(G=G,mpfp=mpfp,level=level,couples=[[1,2],[3,4]])  -- Choice of Cross-cut
    axes = plotLSS(G=G,mpfp=mpfp,level=level,listG=[G1,G2], listOpt=[{'colors': 'red'},{'colors': 'blue'}])  -- Automatically set listLevel to level. Other fields set to optDefault [2 additional functions, same level]
    axes = plotLSS(G=G,mpfp=mpfp,level=level,listG=[G1], listOpt=[{'colors': ['red','blue']}], listLevel=[-295,-290],save=True) -- 1 additional function, for 2 levels. Careful with listOpt fields. listLevel with increasing order!!
    
    -- Case: Coloring with G. Then add stuffs
    axes1 = plotLSS(G=G,mpfp=mpfp,level=level, show=False, save=False)
    axes2 = plotLSS(mpfp=mpfp,level=level,listG=[G1], listOpt=[{'colors': ['red','blue']}], listLevel=[-295,-290],save=True, axes=axes1)

    -- Input: 
    G                      : Principal Function Coloring Safe/Failure domains [(N,d)-> (N,)]
    mpfp                   : (d,) - reference point for cross cut
    level                  : used to color safe/fail G, and level
    axes                   : f, axes = plt.subplots() object [Optional]
    nCol,nRaw              : Optional - subplot dimension
    couples                : list of (2,) arrays - list of cross-cut dimensions
    colorSafe/Fail         : Coloring G
    a,b,Npts               : Npts**2 used, x,y=linspace(a,b,Npts) -> meshgrid
    centered,c             : if True, plot centered [MPFP+-c] 
    x,yTicks               : List of Values
    optG                   : Dict options contour G=level
    listG**                : list of Additional functions
    listLevel**            : Additional levels. Default: [level]
    listOpt**              : list of options contour listG[p]=listLevel[p], initialized with optDefault

    -- Output: 
    axes                   : f, axes = plt.subplots() object 
    """
    # Import
    from pylab import cm
    from matplotlib import colors
    import math
    plt.rc('font', weight='bold',size=fontsize)

    # Initialization
    d = len(mpfp)
    if couples is None: couples = [[i,j] for i in range(d) for j in range(d) if j>i]
    if level is None:
        level = 0; print('Level not defined: set to 0')
    nC = len(couples)
    if nCol is None: nCol = math.ceil(np.sqrt(nC))
    if nRaw is None: nRaw = math.ceil(nC/nCol)
    if axes is None: _, axes = plt.subplots(nRaw,nCol, figsize=(figsize,nRaw/nCol*figsize))
    axes = axes.flatten()
    if xTicks is None: xTicks=[a,b]
    if yTicks is None: yTicks=[a,b]
    nG = len(listG)
    if len(listOpt)==0:        listOpt = [deepcopy(optDefault) for i in range(nG)]
    if len(listLevel) ==0:     listLevel = [level]
    for i in range(nG): listOpt[i] = dictBuildDepth(optDefault, listOpt[i])
    
    # Other Init
    tabMPFP = repeat1D(mpfp,Npts**2)
    cmap = colors.ListedColormap([colorFail, colorSafe])
    bounds=[-1000,0,1000]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.tight_layout()

    # Function Evaluation
    def f(fun=None, i=None, j=None):
        global xmin, xmax
        if centered:
            xmin = [min(mpfp[i]-c,0),min(mpfp[j]-c,0)]
            xmax = [max(mpfp[i]+c,0),max(mpfp[j]+c,0)]
        else:xmin = [a,a] ; xmax = [b,b] 
        x                  = np.linspace(xmin[0], xmax[0], Npts)
        y                  = np.linspace(xmin[1], xmax[1], Npts)
        x,y                = np.meshgrid(x,y)
        u = tabMPFP.copy()
        u[:,i] = x.reshape(-1)
        u[:,j] = y.reshape(-1)
        z                  = fun(u)
        z                  = z.reshape(x.shape)
        a_colors = z.copy()
        a_colors[z<level] = -10
        a_colors[z>level] = 10
        return x, y, z, a_colors
    
    # -----
    # LOOP
    # -----
    for ii in range(nRaw):
        for jj in range(nCol):
            k = jj+ii*nCol
            # Case large k
            if k >= nC:
                axes[k].axis('off')
                continue

            # Basic Plot
            i,j = couples[k]
            axes[k].set_title('(%s,%s)'%(i+1,j+1),fontweight="bold")
            axes[k].scatter(mpfp[i],mpfp[j],color='black',s=100,marker='s',zorder=100)
            axes[k].scatter(0,0,color='black',s=100,marker='+',zorder=100)
            axes[k].set_xticks(xTicks)
            axes[k].set_yticks(yTicks)

            # G coloring + contours
            if G is not None:
                x, y, z, a_colors=f(fun=G,i=i,j=j)
                axes[k].imshow(a_colors, origin='lowest', extent=[x.min(), x.max(), y.min(), y.max()], interpolation='nearest', aspect='auto',cmap=cmap,zorder=0)
                axes[k].contour(x, y, z, [level], **optG)

            # New Levels
            for p in range(nG):
                x,y,z,_=f(fun=listG[p],i=i,j=j)
                axes[k].contour(x, y, z, listLevel, **listOpt[p])
                
    # Output
    if save:
        print('-- Saved in %s'%(fileSave))
        plt.savefig(fileSave, dpi=dpi)
    if show:        plt.show()
    plt.close()
    return axes







def plotArraySimple(x,save=False, show=True, fileName='multi.png',dpi=300,s=1,color='black',**opt):
    """
    Subplot of array x in 2D projections - Simple
    -- Tutorial
    plotArraySimple(x)

    Input:
    -> opt [Dictionary of other options for scatter]
    """
    # Init
    n,d = x.shape
    opt.update({'s':s,'color':color})

    # Double Loop
    fig,axes = plt.subplots(nrows=d,ncols=d,sharex=True,sharey=True,squeeze=True)
    for i in range(d):
        for j in range(d):
            # plt.subplot(d,d,i+j*d+1)
            # for spine in plt.gca().spines.values():
            #     spine.set_visible(False)
            axes[i,j].scatter(x[:,i],x[:,j],**opt)
            
            # plt.xticks([])
            # plt.yticks([])

    # End Plot
    if save: plt.savefig(fileName,dpi=dpi)
    if show: plt.show()
    return

def plotArray(x=None, color=None, marker=None, size=None, label=None,xlabel=None,
              save=False, show=True, fileName='multi.png',dpi=300, normalize=False,
              left=.1, bottom=.1, right=.99, top=.99,
              alpha=0.3,i_PDF=0,**opt):
    """
    Subplot of array x in 2D projections, with KDE at diagonal
    -- Tutorial
    plotArray([0.9*x+0.1,x],color=['red','blue'],label=['Data','New'],marker=['s','d'],size=[5,5],normalize=False,save=True, i_PDF=1)

    -- Input:
    x                   : List of nc (ni,d) arrays
    color               : List of nc Strings
    marker,size,label   : List of nc markers,size for scatter plots
    xlabel              : List of d strings (for each variable)
    alpha               : Transparency fill PDF
    i_PDF               : x[i_PDF][:,k] is considered for 1D PDF
    left,...,top        : margins in %%
    """
    # Init
    assert x is not None
    nc = len(x)
    try: assert len(color)==nc
    except: color = [colorList[i] for i in range(nc)]
    try: assert len(marker)==nc
    except: marker = [None for i in range(nc)]
    try: assert len(size)==nc
    except: size = [1 for i in range(nc)]
    try: assert len(label)==nc
    except: label = ['Data_%s'%(i+1) for i in range(nc)]

    # Find xmin,xmax
    _,d = x[0].shape
    xmin  = [np.array([x[k][:,i].min() for k in range(nc)]).min() for i in range(d)]
    xmax  = [np.array([x[k][:,i].max() for k in range(nc)]).max() for i in range(d)]
    
    # Normalizer
    def norm(xx,aa,bb):
        n,d = xx.shape
        xxx = np.zeros(xx.shape)
        for k in range(d):
            a = aa[k] ; b = bb[k]
            xxx[:,k] = (xx[:,k]-a)/(b-a)
        return xxx
    if normalize:
        x = [norm(x[i],xmin,xmax) for i in range(nc)]
        xmin=np.zeros(d) ; xmax = np.ones(d)
    try: assert len(xlabel)==d
    except:  xlabel = [r'$x_%s$'%(k+1) for k in range(d)]
    
    # KDE 1D
    from statsmodels.nonparametric.kde import KDEUnivariate 
    def f(xx,a,b):
        kde = KDEUnivariate(xx) ; kde.fit()
        # s = np.linspace(a,b,100)
        # e = kde.evaluate(s)
        s = kde.support; e=kde.density
        e = (b-a)*e/e.max() + a
        return s, e

    # Double Loop
    fig,axes = plt.subplots(nrows=d,ncols=d,sharex='col',sharey='row',squeeze=False,figsize=(8,8))
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    mem=True
    for i in range(d):
        for j in range(d):
            if j>i:
                axes[i, j].axis('off')
                continue
            if j==i:
                s,e=f(x[i_PDF][:,i],xmin[i],xmax[i])
                axes[i,i].fill_between(s, e,e.min(), facecolor=color[i_PDF], alpha=0.3)
                axes[i,i].plot(s,e,lw=2,color=color[i_PDF])
                # axes[i,i].set_aspect('equal')

            if j<i:
                for k in range(nc):
                    u = x[k][:,i] ; v = x[k][:,j]
                    opt = {'color':color[k],'marker':marker[k],'s':size[k]}
                    if label[k] is not None and mem: opt['label'] = label[k]
                    axes[i,j].scatter(v,u,**opt)
                    # axes[i,j].set_aspect('equal')
                mem=False
                
    # Legend
    l = fig.legend(loc='upper right',fontsize=25)
    for k in range(len(l.legendHandles)):
        l.legendHandles[k]._sizes = [30]

    # x,y labels
    [plt.setp(axes[-1, k], xlabel=xlabel[k]) for k in range(d-1)]
    [plt.setp(axes[k, 0], ylabel=xlabel[k]) for k in range(1,d) ]

    # End Plot
    if save: plt.savefig(fileName,dpi=dpi)
    if show: plt.show()
    return

def plotPDF1D(x,N=200,color='green'):
    """ Fast Plot PDF 1D: 1D list or (n,) array"""
    # Init
    try:    xmin = x.min()
    except: x = np.array(x)
    a = x.min() ; b = x.max() ; n = len(x)
    # from statsmodels.nonparametric.kde import KDEUnivariate 
    import statsmodels.api as sm
    
    # KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()
    # kde = KDEUnivariate(x) ;
    # s = np.linspace(a,b,N)
    # e = kde.evaluate(s)
    mu = x.mean()
    sig = x.std()
    s = kde.support; e=kde.density
    
    # Plot PDF
    fig,ax = plt.subplots()
    ax.fill_between(s,e,0,facecolor=color,alpha=0.3)
    ax.plot(s,e,lw=2,color=color)
    ax.scatter(x,np.ones(n)*-0.02*(e.max()-e.min()),color='red',marker='+',s=5,label='Realizations')
    plt.show()

    # Fast Histogram
    plt.hist(x)
    plt.show()
    
    return
    


# END--------------PLOT-------------------------


# -----------------ERROR HANDLING---------------
# Return name of variable as string
def getVarName(**kwargs):
    """
    Return name of variable as string
    CAll : getVarName(myVar=myVar)
    returns 'myVar'
    """
    return kwargs.keys()[0]

# Return line in code
def line_code():
    return currentframe().f_back.f_lineno
# end function line_code

# Return filename and line, from anywhere it is called
def fileLine_code(basic=False):
    
    caller = getframeinfo(stack()[2][0])    # 1 step upper
    if basic:
        caller = getframeinfo(stack()[1][0])    # basic
    return (caller.filename, caller.lineno)
# end function fileLine_code

# -------------
# Message Containing FileName and Line code, from where it is called
# -> Possible to put pre/post string to this message
# -> Message = pre + preFile + "File: ...\n" + preLine +"Line: ...\n" + post
# -------------
# message = fileLine_message(pre = "Error!!\n", post = "...Please modify input.", preFile = "     ->", preLine = "   :")
def fileLine_message(**opt):
    dictDefault     = {"pre": "", "post": "", "preFile": "-> ", "preLine": "-> "}
    dictOut         = dictBuildDefault(dictDefault, **opt)
    pre             = dictOut["pre"]
    post            = dictOut["post"]
    preFile         = dictOut["preFile"]
    preLine         = dictOut["preLine"]
    caller          = getframeinfo(stack()[1][0])
    message         = pre + preFile + "File: "+ caller.filename + "\n" + preLine + "Line: " + str(caller.lineno) + "\n" + post
    return message
# end function fileLine_message


# -------------
# Print Message related to i/o Error
# -> if "r": Reading Error. Print file where it is called, and which line
# -------------
# message = message_ioError(f,l,"r",fileName, show = True)
def message_ioError(filePy,line,readOption,fileName, **opt):
    """
Print Message related to i/o Error
-> if "r": Reading Error. Print file where it is called, and which line
-------------
message = message_ioError(f,l,"r",fileName, show = True)

    Input:
    -> filePy       : string - File python from where error is called
    -> line         : integer- Line Python File
    -> readOption   : "r","w"or "c" for read/write/close
    -> fileName     : string - file trying to be read
    """
    # Options
    dictDefault     = {"show": True}
    dictOut         = dictBuildDefault(dictDefault, **opt)
    show            = dictOut["show"]
    mR              = {"r": "Reading", "w": "Writing", "c": "Closing"}
    
    # Message
    message         = []
    message.append("\n")
    message.append("ERROR " + mR[readOption] + " file: " + fileName + "\n")
    message.append("  -> Called in Python file: "+ filePy + "\n")
    message.append("  ->                  line: "+ str(line) + "\n")
    message.append("  -> Type Ctr+D to keep on running code at your own risk :)\n")
    message         = ''.join(message)

    # Print
    if show:
        print(message)
    
    return message
# end function message_ioError

def printDict(d):
    """d=Dictionary To print"""
    m = ''
    for k,v in d.items():
        m+='%s: %s\n'%(k,v)
    print(m)
    return m

# -------------
# Print list as string
# -------------
def printList(l, separator=' '):
    """
    # Print list as string ; separator to separate
    """
    n_l = len(l)
    m = ''
    for i in range(n_l):
        if l[i] in ['',None]: continue
        m += '%s%s'%(l[i],separator)
    return m
    
# -------------
# Create Message containing x array
# -------------
def printTab(x, init='', end='', initLine='x_', initVect=' = [', endVect=']', separator=', ', showLine=True, f=[2,4], scientific=False):
    """
    message = printTab(x, init = '   :')
    Input:
    -> x            : (N,d) array
    """
    # Size
    if len(x.shape) == 1: x = x.reshape(1,-1)

    # Initialization
    (n,d)           = x.shape
    message         = "%s"%(init)
    nSep            = len(separator)
    
    # Loop Rows
    for i in range(n):

        # New Line
        message  += initLine
        if showLine: message += '%s'%(i+1)
        message  += initVect

        # Loop Columns
        for j in range(d):
            if scientific: 
                message     += "%+*.*E%s"%(f[0],f[1],x[i,j],separator)
            else:
                message     += "%+*.*f%s"%(f[0],f[1],x[i,j],separator)



            # message     += "%7.7e%s"%(x[i,j],separator)
        message = message[:-(nSep)]
        message  += '%s\n'%(endVect)

    message = message[:-1]
    message += end
    return message
# end function printTab


# -------------
# Create Message containing x array
# -------------
def printTab1D(x, init='[', end=']', separator=', ', f=[2,4], scientific=False):
    """
    message = printTab1D(x,init='[',end=']')
    Input:
    -> x            : (N,) array
    -> f            : format (2.4) -> %+2.4f or %+2.4E [if Scientific]
    """
    # Initialization
    try: n               = len(x)
    except:
        x = [x] ; n = 1
    m               = "%s"%(init)
    nSep            = len(separator)

    # Loop Rows
    for i in range(n):
        if scientific: 
            m     += "%+*.*E%s"%(f[0],f[1],x[i],separator)
        else:
            m     += "%+*.*f%s"%(f[0],f[1],x[i],separator)
    m = m[:-(nSep)]
    m += end
    return m
# end function printTab1D


# Renumerotation Indice 1D
def indiceRenumerotation(N=0, offset=0, inverse=[], remove=[], order=['remove','offset','inverse']):
    """
    Indice Renumerotation 1D: offset (roll i), inverse: i<->j, remove: i:j(included) with user order
    Input: 
    -> N                    : ind=arange(N) = input set
    -> offset               : Integer. offset=1, N=6 => [6,0,1,2,3,4]
    -> inverse              : [i,j] => ind[i] <-> ind[j] 
    -> remove               : [i,j] => delete ind["i:j" included]
    -> order                : list of max 3 Strings in ['remove','offset','inverse']. order[0], ..., order[2] done
    Output:
    -> ind                  : (N,) integer array - permutation in N
    Ex: 
    indiceRenumerotation(N=10,offset=2,order=['offset'])          [8,9,0,...,7]
    indiceRenumerotation(N=10,remove=[4,6],order=['remove'])      [0,1,2,3,7,8,9]
    indiceRenumerotation(N=10,remove=[4,6],inverse=[2,5],order=['remove','inverse'])      [0,1,2,3,7,8,9] -> [0,1,8,3,7,2,9]
    """
    # Initialization
    ind0  = np.arange(N)

    # Elementary Function Definition: f[k](arg=arg, ind=ind0)
    def f_offset(arg=None,ind=None):
        return np.roll(ind,arg)
    def f_inverse(arg=None,ind=None):
        if len(arg)==0: return ind
        a = ind.copy() ; a[arg[0]] = ind[arg[1]] ; a[arg[1]] = ind[arg[0]]
        return a
    def f_remove(arg=None, ind=None):
        if len(arg)==0: return ind
        indR = np.arange(arg[0],arg[1]+1)
        res = np.delete(ind,indR)
        return res
    f   = {'remove': f_remove, 'offset': f_offset, 'inverse': f_inverse}
    ARG = {'remove': remove, 'offset': offset, 'inverse': inverse}
    # Evaluation
    for k in range(len(order)):
        ok = order[k]
        ind0 = f[ok](arg=ARG[ok],ind=ind0)
    return ind0
# End Function indiceRenumerotation

# Renumerotation Indice 1D
def indiceRenumerotationReverse(N=0, offset=0, inverse=[], remove=[], order=['remove','offset','inverse']):
    """
    Indice Renumerotation 1D - Inverse Operation [remove not considered]. offset, inverse from indiceRenumerotation [read doc]
offset (roll i), inverse: i<->j, remove: i:j(included) with user order
    -> ind                  : (N,) integer array - permutation in N
    Ex: 
    indiceRenumerotation(N=10,offset=2,order=['offset'])          [8,9,0,...,7]
    indiceRenumerotation(N=10,remove=[4,6],order=['remove'])      [0,1,2,3,7,8,9]
    indiceRenumerotation(N=10,remove=[4,6],inverse=[2,5],order=['remove','inverse'])      [0,1,2,3,7,8,9] -> [0,1,8,3,7,2,9]
    """
    remove=[]
    order = [order[-k] for k in range(len(order)) if k != 'remove']
    return indiceRenumerotation(N=N,offset=-offset,inverse=inverse,remove=remove,order=order)
# End Function indiceRenumerotationReverse


# Generate Line/Semi-Circle
def generateEdgeCircle(A,B,Npts, linear=False):
    """
    Input: 
    -> Npts         : integer - Number of points
    -> A            : (2,) array - First  Point
    -> B            : (2,) array - Second Point
    -> linear       : If true, linear instead of semi-circle
    Output:
    -> M            : (Npts,2) array - semi-circular set of points, not includes A, B
   """
    # I, r
    A = np.array(A) ; B = np.array(B)
    I               = (A+B)/2.0
    r               = np.sqrt(np.sum((A-I)**2))

    # {cos,sin}_phi
    phi             = (A-I)/r
    c_phi           = phi[0]
    s_phi           = phi[1]

    # Initialization t, M (remove A,B)
    t               = np.linspace(0, np.pi, Npts+2)
    t               = t[1:-1]
    M               = np.random.random((Npts,2))*0

    # Computation
    ct              = np.cos(t)
    st              = np.sin(t)
    M[:,0]          = I[0] + r*(ct*c_phi - st*s_phi)
    M[:,1]          = I[1] + r*(ct*s_phi + st*c_phi)

    # If Linear, straight line
    if linear:
        M  = np.zeros((Npts,2))
        for k in range(2):
            M[:,k]  = np.linspace(A[k], B[k], Npts+2)[1:-1]
    
    return  M
# End Function generateEdgeCircle



# -------------
# Convert Time from s, to j h m s
# -------------
# message = convertTime(t)
def convertTime(t0):
    """
    Input:
    -> t            : float (s)
    """
    # dict, extract integer seconds
    dictTime        = {'d': 86400, 'h': 3600, 'm': 60, 's': 1}
    t               = int(t0)
    if t < 2: return '%1.4f s'%(t0)
    
    # Function Conversion
    def inConv(tin, name):
        t_temp          = tin%dictTime[name]
        num             = (tin - t_temp)/dictTime[name]
        return (num, t_temp)

    # Function Message
    def inMess(val, name):
        if name == 'd':
            m               = '%d%s '%(val,name)
        else:
            m               = '%02d%s '%(val,name)
        if val == 0:
            return ''
        return m

    # days, hours, minutes, seconds
    (d,t)           = inConv(t, 'd')
    (h,t)           = inConv(t, 'h')
    (m,t)           = inConv(t, 'm')
    (s,t)           = inConv(t, 's')

    # message
    message         = ''
    message        += inMess(d,'d') 
    message        += inMess(h,'h') 
    message        += inMess(m,'m') 
    message        += inMess(s,'s') 

    # Case less than 0s
    if message == '':
        message = '0s '
    
    
    return message[:-1]
# end function convertTime

# Foot,Inch to meter
def convertInch(f,i):
    """
    meter = convertInch(foot,inch)
    """
    return 0.02540000081279999988*i + 0.3048000097536*f
# END--------------ERROR HANDLING---------------



# Numerical Gradient Computation
def gradNumFunction(f, h = 1e-4, d = None, p = None):
    """
    Numerical Gradient Function, given a function f: (n,d) -> (n,p)

    f:   (n,d) array -> (n,p) array - xi = x[i,:] i-th point, in dimension d ; f(xi) = f[i,:] in dimension p
    gf:  (n,d) array -> (n,d,p) array - gf[i,:,:] is the pxd jacobian matrix of f evaluated at xi 
    Call: gf = gradNumFunction(f, h = 1e-4, d = 3, p = 5)


    Input: 
      -> f                  : function
      -> h                  : Float Numerical derivative value
      -> d,p                : f: (n,d) -> (n,p) or (d,) -> (p,) 
    Output:
      -> gf                 : fun: (n,d) -> (n,p,d) or (d,) [or (1,d)] -> (p,d) Gradient
    """
    # Define Function Gradient, depending on f and h, for x = (d,) or (1,d) array
    def gf_unit(x):
        """
        x:      (d,) or (1,d) array
        grad:   (p,d) array
        """
        # Init sizes
        if len(x.shape) == 1:
            x = x.reshape(1,d)
        g = np.zeros((p,d), float)
        
        # Initialize x_up, x_down
        x_up         = np.copy(x)
        x_down       = np.copy(x)

        # Loop
        for j in range(d):
            x_up[0,j]    += h/2.0
            x_down[0,j]  -= h/2.0
            f_up          = f(x_up).reshape(-1) 
            f_down        = f(x_down).reshape(-1)
            x_up[0,j]    -= h/2.0
            x_down[0,j]  += h/2.0
            g[:,j]        = (f_up-f_down)/h
        return g


    # Define Function Gradient, depending on f and h, for x = (n,d) 
    def gf(x):
        """
        x:      (n,d) array
        grad:   (n,p,d) array
        """
        # Init sizes
        if len(x.shape) == 1:
            x = x.reshape(1,d)
        (n,_) = x.shape
        grad  = np.zeros((n,p,d), float)

        # Loop
        for k in range(n):
            grad[k,:,:] = gf_unit(x[k,:])

        # Output
        if n == 1:
            return grad[0,:,:]
        else:
            return grad

    # Return Gradient function
    return gf

# End Function gradNumFunction



# ------------------------------------------------------
# Cluster Functions - Running Automatic Batch
# ------------------------------------------------------
# ---------
# Error Functions
# ---------
def mySystem(command,message=None):
    if message is None:
        message = 'Error while performing:\n%s\n'%(command)
    try:    assert os.system(command) ==0
    except: checkError(False,message=message)
    return

def checkError(b_bool,message=''):
    """
    If b_bool False, error!
    """
    if b_bool: return 0
    m = 'Error: %s\nExit -1\n'%(message)
    print(m)
    sys.exit()
    return 
    
def checkErrorFile(fileName,message='',stop=True):
    """
    If File does not exist -- Message + exit
    """
    if os.path.exists(fileName): return 0
    m = 'Error: %s does not exist.\n%s\nExit -1\n'%(fileName,message)
    print(m)
    if stop: sys.exit()
    return -1

def eliminateComment(fName):
    """
    -- data =readString(fName)
    d = eliminateComment(fName), list of string
    data = list of string. 

    Take off '\n'
    Everything after a comment is removed - Every empty line is removed
    Add "\n" after each line
    """
    data = readString(fName)
    n = len(data)

    # Remove \n
    d1 = [data[k].replace('\n','') for k in range(n)]

    # Remove Empty Lines
    d = [d1[k] for k in range(n) if len(d1[k])>0]
    n = len(d)
    # Loop
    ind= [i for i in range(n) if '#' not in d[i][:1]]
    for k in ind:
        d[k] = d[k].split('#')[0]
    d = [d[k]+'\n' for k in range(n)]
    return d

def initScript():
    """Init Script: go to directory where file is"""
    m ='#!/bin/bash -l\n\n'

    m+='# Folders\n'
    m+='DIR_INIT=$PWD\n'
    m+='DIR_FILE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\n'
    m+='FILENAME=$(readlink -f ${0})\n\n'

    m+='# -- Go to case folder\n'
    m+='cd $DIR_FILE &>/dev/null\n\n'

    return m

def endScript():
    """End Script: go back to initial directory"""
    m='# -- Success Exit - Back to Initial Directory\n'
    m+='cd $DIR_INIT &>/dev/null\n'
    m+='exit 0\n'
    return m

def runScript(command=None,fileScript=None,prepareOnly=False,parallel=False):
    # Prepare Script
    assert command is not None and fileScript is not None
    m = initScript()
    m+= command
    m+= endScript()

    # Save-Make exe-run
    writeString(fileScript,m)
    mySystem('chmod +x %s '%(fileScript))
    if prepareOnly: return fileScript
    if parallel: fileScript='%s &'%(fileScript)
    mySystem(fileScript)
    return 



# -------------
# Prepare batch file
# -------------
def createBatch(folder=None, jobName='job', output='log', ncpuBatch=1, nhours=1, nminutes=0, exeFile='run.sh', NCPUMAX=100, NHOURSMAX=1000, codeExe=None, createFolder=False):
    """
    Prepare batch file in folder: 
    -> job, ncpuBatch, nhours, nminutes, exeFile

    Call:
    fileBatch = createBatch(folder='.../x_0/', ncpuBatch=24, nhours=10, exeFile='run.sh')

    Input:
    -> folder                 : String - Default = './' [absolute address]
    -> jobName                : String - Name of job, when squeue
    -> output                 : String - Stdout, where output is written
    -> ncpuBatch                  : Integer - Number of procs (mpi tasks) (Maximum = 100)
    -> nhours                 : Integer - Number of Hours (Maximum = 1000)
    -> nminutes               : Integer - Number of Minutes
    -> exeFile                : String - Name of Executable to run: e.g:
run.sh 24
    -> codeExe                : String - if given, no exeFile, just codeExe at the end of the file
    -> createFolder           : Bool - If True and folder do not exists, create it

    Content Example batch:
#!/bin/bash
#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 24                 # Total number of mpi tasks requested
#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours

echo "Running [your app] with: $cmd on $SLURM_JOB_NODELIST in directory "`pwd`
date
run.sh 24
    """
    # Initialization
    if folder is None:
        folder      =  os.getcwd() + '/'
    if folder[-1] != '/': folder = folder + '/'
    if createFolder: os.system('mkdir -p %s'%(folder))
    fileScript = '%sbatch'%(folder)

    # Check nhours/ncpuBatch
    nhours     = int(nhours+0.1)
    ncpuBatch      = int(ncpuBatch+0.1)
    nminutes   = int(nminutes)%60
    if nhours > NHOURSMAX:
        print('--> In folder: %s\nnhours = %s > %s => nhours = %s'%(folder,nhours,NHOURSMAX,NHOURSMAX))
        nhours     = NHOURSMAX
    if ncpuBatch > NCPUMAX:
        print('--> In folder: %s\nncpu = %s > %s => ncpu = %s'%(folder,ncpuBatch,NCPUMAX,NCPUMAX))
        ncpuBatch     = NCPUMAX

    # Content of batch file
    m          = '#!/bin/bash\n\n'
    m         += '#SBATCH -J %s               # Job name \n'%(jobName)
    m         += '#SBATCH -o %s               # Name of stdout output file expands to jobId)\n'%(output)
    m         += '#SBATCH -n %s                 # Total number of mpi tasks requested\n'%(ncpuBatch)
    m         += '#SBATCH -t %s:%02d:00           # Run time (hh:mm:ss)\n\n'%(nhours,nminutes)
    m         += 'echo "Running [your app] with: $cmd on $SLURM_JOB_NODELIST in directory "`pwd`\ndate\n\n'

    # Executable
    if codeExe is None:
        m         += '%s %s\n'%(exeFile,ncpuBatch)
    else:
        m         += codeExe + '\n'

    # Write in fileScript
    with open(fileScript, 'w') as fout:
        fout.writelines(m) 

    return fileScript
# End Function createBatch
# -------------


# -------------
# Prepare batch file
# -------------
def createBatchParallel(fileBatch=None, jobName='job', output='log', nnodes=1, ncpuBatch=24, ncpuExe=1, nhours=1, nminutes=0, exeFile='run.sh', indParallel=[], preFolder='', NCPUMAX=100, NHOURSMAX=1000, iBatch=''):
    """
    Prepare batch file in folder for parallel Run: 

    Call:
    fileBatch = createBatch(fileBatch='..../batch_0', jobName='job_0', output='log_0', ncpuBatch=24, ncpuExe=4, nhours=10, exeFile='run.sh', indParallel=[0,1,2,3,4,5], preFolder='../SU2/t_')

    Input:
    -> fileBatch              : String - where to store the batch file, address should be valid
    -> jobName                : String - Name of job, when squeue
    -> output                 : String - Stdout, where output is written
    -> ncpuBatch             : Integer - Number of procs (mpi tasks) [24 or 32]
    -> ncpuExe               : Integer - Number of procs for single run [4]
    -> nhours                 : Integer - Number of Hours (Maximum = 1000)
    -> nnodes                 : Integer - Number of Nodes
    -> nminutes               : Integer - Number of Minutes
    -> exeFile                : String - Name of Executable to run: e.g:
run.sh 24
    -> indParallel            : list of integers --> folder[batch]/preFolder$[ind[i])/run.sh $ncpuExe
    -> preFolder              : see above, address relative from zhere batch is created
    -> iBatch                 : END_BATCH$i if not None, for end of batch
    Content Example batch:
#!/bin/bash
#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                  # Total number of nodes requested
#SBATCH -n 24                 # Total number of mpi tasks requested
#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours

echo "Running [your app] with: $cmd on $SLURM_JOB_NODELIST in directory "`pwd`
date
run.sh 24
    """
    # Check nhours/ncpuBatch
    nhours     = int(nhours+0.1)
    ncpuBatch      = int(ncpuBatch+0.1)
    nminutes   = int(nminutes)%60
    if nhours > NHOURSMAX:
        print('--> In folder: %s\nnhours = %s > %s => nhours = %s'%(folder,nhours,NHOURSMAX,NHOURSMAX))
        nhours     = NHOURSMAX
    if ncpuBatch > NCPUMAX:
        print('--> In folder: %s\nncpu = %s > %s => ncpu = %s'%(folder,ncpuBatch,NCPUMAX,NCPUMAX))
        ncpuBatch     = NCPUMAX

    # Intro
    m = '#!/bin/bash\n\n'
    m+= '# -----------------------------\n'
    m+='# Date:   %s, %s\n'%(time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S"))
    m+='# Author: Nassim Razaaly\n'
    m+='# Script: batch to be submitted from the current folder!\n'
    m+='#         --> Run SU2 Parallel\n'
    m+='# -----------------------------\n\n'

    # Content of batch file
    m         += '#SBATCH -J %s               # Job name \n'%(jobName)
    m         += '#SBATCH -o %s               # Name of stdout output\n'%(output)
    m         += '#SBATCH -N %s                 # Total number of mpi tasks requested\n'%(nnodes)
    m         += '#SBATCH -n %s                 # Total number of mpi tasks requested\n'%(ncpuBatch)
    m         += '#SBATCH -t %s:%02d:00           # Run time (hh:mm:ss)\n\n'%(nhours,nminutes)
    m         += 'echo "Running [your app] with: $cmd on $SLURM_JOB_NODELIST in directory "`pwd`\ndate\n\n'

    # Parameters
    m+='# -- Parameters:  Prefolder= Relative Address from current folder, where file is defined.\n'
    m+='PRE_FOLDER=%s\n'%(preFolder)
    m+='NCPU=%s\n\n'%(ncpuExe)

    # Example
    m+='# # -- Loop in Comment: Fast Test\n'
    m+='# N_TOT=50\n'
    m+='# for n in $(seq 1 $N_TOT)\n'
    m+='# do\n'
    m+='#    echo ${PRE_FOLDER}$(($n-1))/run.sh $NCPU &\n'
    m+='# done\n'
    m+='# wait\n\n'

    # Loop Command
    m+='# -- Run in Parallel\n'
    for i in indParallel:
        m+='${PRE_FOLDER}%s/%s $NCPU &\n'%(i,exeFile)
        m+='sleep 1\n'
    m+='\nwait\n'

    # Exit Success
    m+='# -- Success Exit\n'
    m+='date\n'
    m+='touch END_BATCH%s\n'%(iBatch)
    m+='exit 0\n'

    # Write in fileScript
    folder = os.path.dirname(os.path.realpath(fileBatch))
    if os.path.exists(folder) is False: os.system('mkdir -p %s'%(folder))
    with open(fileBatch, 'w') as fout:
        fout.writelines(m) 

    return fileBatch
# End Function createBatchParallel
# -------------


# -------------
# Prepare script for run batchs
# -------------
def globalBatchScript(listBatch=[], folder=None, fileScript=None, localRun=False, batchDir=None, wait=True):
    """
    Prepare a Script for running a list of batch already given
    
    Input:
    -> listBatch        : String of batch files
    -> folder           : Where to create the script [Default='.']
    -> fileScript       : finalName of script, with folder
    -> localRun         : Bool - if True, folder has relative path: batchDir
    -> batchDir         : Relative path where batches are
    -> wait             : Bool - If True, then wait for batches to be done, to finish.

    Call Example: script = globalBatchScript(listBatch=listBatch, folder='....', fileScript='global.sh')
    or script = globalBatchScript(listBatch=listBatch, folder='....', fileScript='global.sh', localRun=True, batchDir='./batch')
--> runAll.py:
import generalFunctions as gF
import os
ncpuBatch=24
nhours=10
exeFile='run.sh'
preLink='.../t_'
preJob  = 't_'
nList=range(200)
listBatch = [gF.createBatch(folder='%s%s/'%(preLink,i), ncpuBatch=ncpuBatch, nhours=nhours, exeFile=exeFile) for i in nList]   
script=gF.globalBatchScript(listBatch=listBatch)  
os.system('chmod +x %s'%(script))
os.system(script)

or 

--> runAll.py: [More General]
import generalFunctions as gF
import os
ncpuBatch=24
nhours=10
exeFile='run.sh'
preLink='.../case/'
foldList=os.listdir(preLink)
nameList=[os.path.basename(f) for f in foldList]
listBatch = [gF.createBatch(folder='%s%s/'%(preLink,f), ncpuBatch=ncpuBatch, nhours=nhours, exeFile=exeFile, jobName=f) for f in nameList]
script=gF.globalBatchScript(listBatch=listBatch)  
os.system('chmod +x %s'%(script))
os.system(script)
   
    Input:
    -> folder                 : String - Default = './' [absolute address] - Where create the script
    -> listBatch              : List of String - full address of batch to run
    """
    # Initialization
    if folder is None:
        folder      =  os.getcwd() + '/'
    if folder[-1] != '/': folder = folder + '/'
    if fileScript is None: fileScript='globalBatchScript.sh'
    script = folder+fileScript

    # Function to find exact END_BATCH for each file batch
    def findBatch(fBatch):
        a = readString(fBatch)
        ind = [i for i, s in enumerate(a) if 'touch' in s]
        return a[ind[0]].split()[1]
    endBatch = [findBatch(listBatch[i]) for i in range(len(listBatch))]

    # Create message
    m          = '#!/bin/bash\n\n'
    m         += '# -------------------\n'
    m         += '# Purpose: Run all batches (see below)\n'
    if localRun: 
        m         += '# --> Local Version: To be run from the current Directory\n'
    m         += '# -------------------\n\n'

    # Initialization
    m+='# -------------------\n'
    m+='# Initialization\n'
    m+='# -------------------\n'
    m+='DIR_INIT=$PWD\n'
    m+='DIR_FILE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\n'
    m+='cd $DIR_FILE\n\n'

    # Loop Submit batches
    m         += '# -------------------\n'
    m         += '# Loop\n'
    m         += '# -------------------\n'
    if localRun: m    += 'cd %s\n\n'%(batchDir)
    for i in range(len(listBatch)):
        s     = listBatch[i]
        DIR   = os.path.dirname(os.path.realpath(s))
        FILE  = os.path.basename(s)
        if localRun == False:        m    += 'cd %s\n'%(DIR)
        m    += 'rm -f %s\n'%(endBatch[i])
        m    += 'sbatch %s\n\n'%(FILE)

    # Loop Wait End Batches
    if wait:
        m+= "while "
        for i in range(len(listBatch)):
            s     = listBatch[i]
            DIR   = os.path.dirname(os.path.realpath(s))
            FILE  = os.path.basename(s)
            m    += '[ ! -f %s/%s ] || '%(DIR,endBatch[i])
        m = m[:-4] + '\n'
        m+= 'do\n'
        m+= '    sleep 10\n'
        m+= 'done\n\n'

    # Get to Initial Folder/Write in script
    m         += '# -------------------\n'
    m         += '# Return to Initial Folder\n'
    m         += '# -------------------\n'
    m         += 'cd $DIR_INIT\n'

    # Write in File
    with open(script, 'w') as fout:
        fout.writelines(m) 
    os.system('chmod +x %s'%(script))

    return script
# End Function globalBatchScript
# -------------


# -------------
# Automatic Run of Batchs
# -------------
def automaticBatch(folder=None, jobName='job_', ncpuBatch=1, nhours=1, nminutes=0, exeFile='run.sh', NCPUMAX=100, NHOURSMAX=1000, codeExe=None, runScript=False):
    """
    Automatic Run of Batchs [from given folder]:
    -> Generates a batch file in each folder
    -> Generate a script for "sbatch" all previous batch
    -> run the script

    Call Example: 
    import generalFunctions as gF
    gF.automaticBatch(folder='.../', jobName='job_', ncpuBatch=1, nhours=1, nminutes=0, exeFile='run.sh', NCPUMAX=100, NHOURSMAX=1000, runScript=False)
       
    Input:
    -> folder                 : String - Default = './' [absolute address] - What folder is considered for batch evaluation
    -> jobName                : String - name=jobName+$subdir, appearing in squeue
    -> ncpuBatch                  : Integer - Number of procs (mpi tasks) (Maximum = 100)
    -> nhours                 : Integer - Number of Hours (Maximum = 1000)
    -> nminutes               : Integer - Number of Minutes
    -> exeFile                : String - Name of Executable to run: e.g:
run.sh 24
    -> codeExe                : String - if given, no exeFile, just codeExe at the end of the file
    -> runScript              : Bool - if True, runs automatically the script
    """
    # Initialization
    if folder is None:
        folder      =  os.getcwd() + '/'
    if folder[-1] != '/': folder = folder + '/'
    preLink    = folder

    # List of Subdirectories [name/full address]
    foldList   = os.listdir(preLink)
    nameList   = [os.path.basename(f) for f in foldList]

    # Create batch in each 
    listBatch  = [createBatch(folder='%s%s/'%(preLink,f), ncpuBatch=ncpuBatch, nhours=nhours, nminutes=nminutes, exeFile=exeFile, jobName=jobName+f, NCPUMAX=NCPUMAX, NHOURSMAX=NHOURSMAX, codeExe=codeExe) for f in nameList]
    script     = globalBatchScript(listBatch=listBatch)  
    os.system('chmod +x %s'%(script))

    # Output/Run Script
    if runScript:    os.system(script)
    out = {'script': script, 'listBatch': listBatch}
    return out
# End Function automaticBatch
# -------------


class ClassBasic:
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        m = '-- Class %s\n'%(self.__class__.__name__)
        for k,v in vars(self).items():
            m+= '%s = %s\n'%(k,v)
        return m
    def __init__(self):
        pass


# Class Template Minimal
class ClassTemplateMin:

    # Default Options
    __opt       = {}

    # Dynamic Properties
    @property
    def folderRoot(self): 
        return getSubDirectory(folder=self.preLink,n=1)

    @property
    def comment(self):
        m = '%s\n'%(datetime.datetime.now())
        try: m+= self.opt['comment']
        except: pass
        m+= self.__str__(full=True)
        try: m+= self.print(show=False)
        except: pass
        return m

    def __init__(self, optParent={}, **opt):
        self.opt               = initDictFull(opt, optParent)
        # self.preLink           = os.path.dirname(os.path.realpath(__file__)) 
        self.listData          = []
        
    def __repr__(self):
        return self.__str__()
        
    def __str__(self,full=True):
        m                    = '----> %s: '%(self.__class__.__name__)
        return m

    def print(self,show=True):
        """
        """
        m = ''
        if show: print(m)
        return m

    def open(self):
        """
        OPEN Folder Template
        """
        os.system('xdg-open %s &'%(self.folder))
        return


# Class Template
class ClassTemplate:

    # Default Options
    __opt       = {}

    # Dynamic Properties
    @property
    def folderRoot(self): 
        return getSubDirectory(folder=self.preLink,n=1)

    @property
    def comment(self):
        m = '%s\n'%(datetime.datetime.now())
        try: m+= self.opt['comment']
        except: pass
        m+= self.__str__(full=True)
        try: m+= self.print(show=False)
        except: pass
        return m

    def __init__(self, **opt):
        self.opt               = initDictFull(opt, ClassTemplate.__opt)
        # self.preLink           = os.path.dirname(os.path.realpath(__file__)) 
        self.listData          = []
        
    def __repr__(self):
        return self.__str__()
        
    def __str__(self,full=True):
        m                    = '----> %s: '%(self.__class__.__name__)
        return m

    def print(self,show=True):
        """
        """
        m = ''
        if show: print(m)
        return m
    
    def save(self, special=None, message=False, fileInput=None):
        """
        --> Save all field in self.listData
        --> special: String (field of self) - if not None, saved in $foldDOE/$special.npy
        --> Save fileInput (generating script)
        self.save() or self.save(special='opt')
        """
        # special
        if special is not None: 
            try: np.save(self.folder+special+'.npy',getattr(self,special))
            except: print('Failed to store self.%s in %s'%(special,self.folder+special+'.npy'))
            return
        
        # FileInput in script
        if fileInput is not None: 
            try:
                a = readString(fileInput)
                a.insert(0,'"""\n%s\n"""\n'%(self.comment))
                writeString(fScript,a)
            except: print('--> fileInput not saved for %s: \n%s'%(field, fileInput))

        # Extract
        d = {}
        for name in self.listData:
            try:    d[name] = getattr(self,name)
            except: pass

        # Save
        np.save(self.fileData,d)
        m = '--> datas %s saved in %s'%(list(d.keys()),self.fileData)
        if message: print(m)

        # Comment
        writeString(self.fileComment, self.comment)
        
        return


    def load(self, special=None, message=False):
        """
        --> Load all field in self.listData
        --> special: String (field of self) - if not None, loaded from $folder/$special.npy to self.$special

        self.load() or self.load(special='opt')
        """
        # special
        if special is not None: 
            try:
                d = np.load(self.folder+special+'.npy',encoding='latin1').item()
                setattr(self,special,d)
                return
            except:
                print('Failed to load from %s to self.%s.'%(self.folder+special+'.npy',special))
                return
            
        # Load Basic
        try:    d = np.load(self.fileData, encoding='latin1').item()
        except:
            print('Load Data from File %s Failed.'%(self.fileData))
            return
        
        # Extract
        for name in self.listData:
            try:    setattr(self,name,d[name])
            except: pass
        m = '--> %s loaded from %s'%(list(d.keys()),self.fileData)
        if message: print(m)
        
        return
    # End Function load
    #--------------------

    def open(self):
        """
        OPEN Folder Template
        """
        os.system('xdg-open %s &'%(self.folder))
        return


# Class Monitor
class Monitor:
    """
    -- Example
    m = Monitor(reset=True, prefix='Fail Proba mu-')
    m('Data')                         : Fail Proba mu-Data: 34.4 s
    m('Gradient')                     : Fail Proba mu-Gradient: 1.236 s

    or 

    m = Monitor()
    m('Data')                         : Data: 34.4 s
    m('Gradient')                     : Gradient: 35.6 s
    m('new Gradient',reset=True)      : new Gradient: 12.3 s
    """
    def __init__(self, reset=False, prefix='', withTime=True):
        self.t0 = time.time()
        self.__t0 = self.t0
        self.prefix=prefix
        self.withTime=withTime

    def totalTime(self):
        return convertTime(time.time()-self.__t0)
        
    def time(self):
        return time.time()-self.t0

    def timeString(self):
        tf = time.time() - self.t0
        return convertTime(tf)

    def reset(self):
        self.t0 = time.time()
    
    def __call__(self,message='', reset=None):
        if reset is None: reset = self.reset
        from datetime import datetime
        if self.withTime:  a = ' -%s'%(datetime.now())
        else: a = ''
        tf = time.time()-self.t0

        if tf < 2:
            print('%s%s: [%s s] %s'%(self.prefix,message,tf,a))
        else:
            print('%s%s: [%s] %s'%(self.prefix,message,convertTime(tf),a))
        if reset: self.reset()

# Class Object Dictionary
class ResultDict:
    """
    Object Easy, from Dictionary
    res = ResultDict(a=1,b=[...],...) -> res.a, res.b...
    opt = res.save(fileSave='file.npy') --> Creation of fileScript = fileLoad.py to load it from ipython automatically
    res.load(opt=opt) or res.load(fileLoad='file.npy')

    -- Recursive Strategy create/load/save
    A = ResultDict(a=2,b=3) ; B = ResultDict(c=-1,A=A)
    o = A.save()
    o = res.save() returns exclusively dictionary


    """
    # def __init__(self, **opt):
    #     [setattr(self,k,v) for k,v in opt.items()]
    #     self._listResultDict=[]

    def __init__(self, **opt):
        self.load(**opt)

    @property
    def dict(self):
        return self.save()
        
    def __str__(self,full=False):
        m = '--> Object %s:\n'%(self.__class__.__name__)
        try: m+= self.comment
        except: pass
        m+= '-- List of Attributes \n'
        for k in self.__dict__.keys():
            m+= '%s\n'%(k)
        
        if full:
            m+= '-- Full List of Attributes [opt] \n'
            for k,v in self.__dict__.items():
                m+= '%s:%s\n'%(k,v)
        return m

    def print(self):
        print(self.__str__(full=True))
        return
    
    def __repr__(self): return self.__str__()


    def script(self, fileSave=None):
        """
        Create script 
        """
        m = '# Import\n'
        m+='import generalFunctions as gF\n'
        m+='import numpy as np\n\n'
        m+='# Load\n'
        m+='opt = np.load("%s").item()\n\n'%(fileSave)
        m+='# Object\n'
        m+='res = gF.ResultDict(**opt)\n'
        m+='print(res)\n'
        return m
        
    def save(self, fileSave=None, fileScript=None):
        """
        if fileSave not None: 
        --> fileSave=$folder/$name.npy
        --> fileScript [if None]: $folder/$nameLoad.py
        """
        # Basic opt
        opt = deepcopy(self.__dict__)
        opt['_listResultDict'] = []
        # Recursive opt
        for k in opt.keys():
            try:
                if opt[k].__class__.__name__=='ResultDict':
                    opt[k] = opt[k].save()
                    opt['_listResultDict'].append(k)
            except:pass
        
        if fileSave is not None:
            if fileSave[-4:] != '.npy': fileSave=fileSave+'.npy'
            np.save(fileSave,opt)

            # fileScript
            if fileScript is None: fileScript = fileSave[:-4]+'Load.py'

            # Script for fileScript
            m = self.script(fileSave=fileSave)
            writeString(fileScript,m)
            m = 'Object %s saved as Dictionary in  %s\n'%(self.__class__.__name__,fileSave)
            m+= 'To Load it, use %s\n'%(fileScript)
            m+= 'Or Simply:\nA = ResultDict() \nA.load(fileScript=%s)\n'%(fileSave)
            print(m)
            
        return opt

    def load(self, _listResultDict=[], **opt):

        for k,v in opt.items():
            if k in _listResultDict:
                setattr(self, k, ResultDict(**opt[k]))
            else:
                setattr(self,k,v)
        
    # def load(self, fileLoad=None, opt=None):
    #     if fileLoad is not None and opt is not None:
    #         print('Load Object %s: fileLoad and opt defined.\nOnly fileLoad will be used.'%(self.__class__.__name__))
    #         opt = None
    #     if opt is not None:
    #         try:
    #             self.__init__(**opt)
    #             print('Object Load Successful from opt')
    #         except:
    #             print('Failed to load Object %s from dictionary opt'%(self.__class__.__name__))
    #             return -1
    #         return
    #     if fileLoad is not None:
    #         try:
    #             opt = np.load(fileLoad).item()
    #             self.__init__(**opt)
    #             print('Object Load Successful from fileLoad=%s'%(fileLoad))
    #         except:
    #             print('Failed to load Object %s from fileLoad=%s'%(self.__class__.__name__,fileLoad))
    #             return -1
    #         return
    #     print('Failed to load Object %s'%(self.__class__.__name__))
    #     return -1
    
# # CLASS OptimObject    
# import scipy.optimize 
# import cma
# import scipydirect
# class OptimObject(ClassTemplateMin):
#     """
#     Example:
#     f: (N,d) -> (N,) array, with possible (d,) -> (1,)
#     xmin,xmax: (d,) arrays
#     A = OptimObject(f=fObj,xmin=xmin, xmax=xmax, optim='CMA',options={...})
#     res = A.solve()
#     res['x'], res['fun'], res['t']
#     """

#     # List Of Options
#     """
#     optim                  : ['CMA','DE','LHS','DIRECT']
#     f                      : Objective Function (N,d) -> (N,) array
#     xmin,xmax              : (d,) arrays
#     x0                     : (1,d) array - Initial Point
#     LHS                    : Options for LHS
#          nLHS                  : Number of Samples
#     CMA                    : Options for CMA
#        -> sigmaCoeff           : INitialization sigma=min(xmax-xmin)*sigmaCoeff
#        -> x0                   : 'Center' -> (self.xmin+self.xmax)/2. Otherwise, if x0 de
#     """
#     __opt = {'optim': 'CMA',
#              'f': None,
#              'xmin': None,
#              'xmax': None,
#              'x0': None,
#              'LHS': {'nLHS': 1e4},
#              'CMA': {'opt': {'tolfun': 1e-3, 'tolx': 1e-5, 'maxfevals': 1e4, 'popsize': 30, 'verb_log': 0, 'verb_disp': 0, 'verbose': -1}, 'sigmaCoeff': 0.95, 'x0': 'Center'},
#              # 'DE': {'opt': {'strategy': 'best2exp', 'popsize': 100, 'polish': True, 'recombination': 0.7, 'tol': 1e-4, 'mutation': 1.0, 'maxiter': 70}, 'restart': 5},
#              'DE': {'opt':{}, 'restart':5},
#              'DIRECT': {'opt': {'maxf': 80000, 'maxT': 80000, 'algmethod': 0, 'eps': 1e-4}}
             
#     }


#     def __str__(self):

#         m = super().__str__()
#         m+= '%s - %sD\n'%(self.opt['optim'],self.d)
#         try: m+= '-> options: %s\n'%(self.opt[self.opt['optim']])
#         except: pass
#         m+= '-> fObj: %s\n'%(self.opt['f'].__doc__)
#         m+= '-> xmin = %s\n'%(printTab1D(self.xmin))
#         m+= '-> xmax = %s\n'%(printTab1D(self.xmax))
#         return m


    
#     @property
#     def xmin(self):
#         return np.array(self.opt['xmin'])

#     @property
#     def xmax(self):
#         return np.array(self.opt['xmax'])
    
#     def options(self,optim=None):
#         """
#         Options according to optim method selected
#         """
#         opt = deepcopy(self.opt)
#         if optim == 'LHS': return opt['LHS']
#         if optim == 'CMA': return opt['CMA']['opt']
#         if optim == 'DE':  return opt['DE']['opt']
#         if optim == 'DIRECT': return opt['DIRECT']['opt']
#         assert 0 == 1
    
#     @property
#     def x0(self):
#         """
#         Starting Point CMA
#         """
#         if self.opt['CMA']['x0'] == 'Center': return (self.xmin+self.xmax)/2
#         if self.opt['x0'] is not None: return self.opt['x0']
#         return (self.xmin+self.xmax)/2

#     @property
#     def sigma(self):
#         """
#         Initial sigma CMA
#         """
#         return self.opt['CMA']['sigmaCoeff']*min(np.abs(self.xmax-self.xmin))
    
#     @property
#     def restart(self): return self.opt['DE']['restart']
    
#     @property
#     def f(self):
#         return self.opt['f']

#     @property
#     def bnds(self):
#         bn = []
#         for k in range(self.d):
#             bn.append(np.array([np.copy(self.xmin[k]), np.copy(self.xmax[k])]))
#         return bn
# #             return  [self.xmin.tolist(),self.xmax.tolist()]

    
#     @property
#     def d(self):
#         return len(self.xmin)

#     @property
#     def optim(self):
#         assert self.opt['optim'] in ['CMA','DIRECT','LHS','DE']
#         return self.opt['optim']
    
#     def __init__(self, **opt):
#         super().__init__(optParent=OptimObject.__opt,**opt)

#     def solve(self, optim=None, options=None, f=None, bnds=None):
#         """
#         Optimization Solve:
#         Output: Dictionary
#         -> x         : (d,) array
#         -> fun       : f(x)
#         -> nfev      : Integer - 'number of f evaluations'
#         -> t         : CPU time
#         """
#         # Initialization
#         if f is None: f = self.f
#         if optim is None: optim = self.optim
#         if options is None: options = self.options(optim=optim)
#         if bnds is None: bnds = self.bnds
#         t0  = time.time()

#         # Switch Case
#         if optim == 'CMA':
#             options['bounds'] = [self.xmin.tolist(),self.xmax.tolist()]
#             res  = cma.fmin(f, self.x0, self.sigma, options = options)
#             out  = {'x': np.array(res[0]), 'fun': res[1], 'nfev': res[2], 't': time.time() - t0}

#         if optim == 'DE':
#             restart = self.restart
            
#             res = [scipy.optimize.differential_evolution(f, bnds, **options) for i in range(restart)]
#             fev = np.array([res[i]['fun'] for i in range(restart)])
#             i   = np.argmin(fev)
#             out = res[i] ; out['t'] = time.time() - t0
            
#         if optim == 'DIRECT':
#             res = scipydirect.minimize(f, bnds, **options)
#             out = {'x': res['x'], 'nfev': options['maxf'], 'fun': res['fun'], 't': time.time() - t0}

#         if optim == 'LHS':
#             nLHS = options['nLHS']
#         return out


    

# -----------------MAIN-------------------------
if __name__ == '__main__':

    # -------------
    # Work Definition
    n_work = ['batchParallel','automBatch','batch']
    n_work = []

    # -------------
    # Batch Test
    # -------------
    # # Sequential
    # if 'batch' in n_work:

    #     preFold = './folder/t_'
    #     preJob  = 't_'
    #     nList   = range(5)
    #     listBatch = [createBatch(folder='%s%s/'%(preFold,i),ncpuBatch=24, nhours=0, nminutes=2, exeFile='run.sh', createFolder=True, jobName='%s%s'%(preJob,i),iBatch=i) for i in nList]
    #     script=globalBatchScript(listBatch=listBatch,iBatch=nList)  

    # Parallel
    if 'batchParallel' in n_work:

        preFold = './folder/t_'
        preJob  = 't_'
        nList   = range(5)
        a = [createBatchParallel(fileBatch='./batch/batch_%s'%(i),ncpuBatch=24, ncpuExe=2,output='log_%s'%(i), nhours=0, nminutes=2, exeFile='run.sh', jobName='job_%s'%(i), preFolder='../SU2/t_', indParallel=[0,1,2],iBatch=i**5+63) for i in nList]
        script=globalBatchScript(listBatch=a)  
        

    # -------------
    # Automatic Batch
    # -------------
    if 'automBatch' in n_work:
        out = automaticBatch(folder='./folder/', jobName='job_', ncpuBatch=1, nhours=1, nminutes=0, exeFile='run.sh', NCPUMAX=100, NHOURSMAX=1000, runScript=False)

    # -------------
    # Word Extraction
    # -------------
    if 'word' in n_work:

        m = 'VARIABLES = "x","y""Density","X-Momentum","Y-Momentum","Energy","Pressure","Temperature","Mach","C<sub>p</sub>"'
#        m = 'VARIABLES = "x""y""Density""X-Momentum""Y-Momentum""Energy""Pressure""Temperature""Mach""C<sub>p</sub>"'
        l = extractWord(m,separator='"')
        print(l)


    # -------------
    # Test Additional Random Points based on Geometric Considerations
    # -------------
    if 'addRandom' in n_work:
        nv = 20 ; d=2
        xv= sampleLHS(nv,np.zeros(d),np.ones(d))
        x = addRandom(xv=xv,K=10,N=1e5)
        scatterPlot(xv,m='o',s=50,c='red')
        scatterPlot(x,m='.',s=20,c='green')
        plt.show()

    # -------------
    # Test Cubic B-splines
    # -------------
    if False: 
        a = 5 ; nTest = 2
        x = np.linspace(0,10,10000) ; y = np.sin(x*a)
        m = Monitor()
        for i in range(nTest):
            f,df = interpSplines1D(x,y,a,a*np.cos(10*a))
            m('Build %s/%s Cubic Splines'%(i+1,nTest)) 
        xx = np.linspace(0,10,55000)
        yy = f(xx)
        m('Evaluate')
        yR = np.sin(xx*a)
        plt.plot(xx,yR,color='red',lw=2,label='Ref')
        plt.plot(xx,yy,color='black',lw=1,label='bSpline')
        plt.legend()
        plt.show()

    # -------------
    # Test BFGS
    # -------------
    if False: 

        def f(x):
            return (x**8).sum() - 3*(x**2).sum()+ x.sum()
        def df(x):
            return 8*x**7 - 6*x + 1

        ff = []
        def rosen(x):
            """The Rosenbrock function"""
            global ff
            y = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
            ff.append(y)
            return y
        
        def rosen_der(x):
            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            der = np.zeros_like(x)
            der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
            der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
            der[-1] = 200*(x[-1]-x[-2]**2)
            return der

        x0 = np.ones(10)*100
        # x0[0]=-1
        from scipy.optimize import minimize
        t0 = time.time()
        res = minimize(rosen, x0, method='BFGS', jac=rosen_der,
                       options={'disp': True,'maxiter':None},)
        print(res)
        print('In ',time.time()-t0)
        print('|Jac|=',np.linalg.norm(res['jac']))

        print('START!')
        r = BFGS(x0=x0,f=rosen,df=rosen_der, NMAX=100, eps=1e-5,modeDebug=True)
        print(r())
        quit
        from newtonSolver import gradNumFunction

        def rosenTab(x):
            if len(x.shape)==1: x = x.reshape(1,-1)
            n = x.shape[0]
            return np.array([rosen(x[k,:]) for k in range(n)])

            
        # dff = gradNumFunction(rosenTab, h=1e-4, d=len(x0), p =1)
        # dff(x0)

        # x0 = np.ones(len(x0))
        # r = BFGS(x0=x0,f=rosenTab, NMAX=3000, eps=1e-9)
        # print(r())


        # code.interact(local=dict(globals(), **locals()))
        
        dfx = r.dfun(r.xSol)
        dfk = r.xSol_grad
        print('||grad(x)-dfx||=',np.linalg.norm(dfx-dfk))
        print('||grad(x)||=',np.linalg.norm(dfx))
        print('||dfk||=',np.linalg.norm(dfk))
        
