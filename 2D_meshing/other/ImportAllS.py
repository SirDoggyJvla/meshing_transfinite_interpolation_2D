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

from cubicSpline import *

