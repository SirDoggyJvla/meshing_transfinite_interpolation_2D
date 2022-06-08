# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:41:46 2022

@author: utilisateur
"""
import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

x1 = np.linspace(0,pi,50) ; y1 = np.sin(x1)*0.2
x2 = -np.linspace(0,2*pi,50) ; y2 = np.sin(x2)*0.05 - 0.2
plt.plot(x1*2-pi,y1)
plt.plot(x2+pi,y2)


