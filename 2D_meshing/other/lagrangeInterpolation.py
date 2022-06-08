# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:06:30 2022

@author: utilisateur
"""
from ImportAllS import *
import sympy as sp

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
