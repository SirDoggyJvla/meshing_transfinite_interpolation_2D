import sys, os, time, inspect
from pyDOE.doe_lhs import lhs
import numpy as np
import time, datetime
import code   #code.interact(local=dict(globals(), **locals()))
import matplotlib.pyplot as plt
from itertools import cycle
import scipy.stats as ss
from six.moves import input
from sklearn.cluster import KMeans
from copy import deepcopy

# HiDPI Matplotlib
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200

import generalFunctions as gF
from math import sin, cos, asin, acos, tan, atan, sqrt, exp, log 
