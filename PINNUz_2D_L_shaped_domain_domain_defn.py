##############################################################################################
###### Packages to Import ######
##############################################################################################
# Standard Maths
import numpy as np
import math
from math import pi, sqrt, exp
# Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
# Plotting & saving data
import os
import csv
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
# Misc
import time
##############################################################################################
###### Standard functions related to the Neural Network ######
##############################################################################################

# Cartesian Grid 
def Domains(N):
        n = 2*N+1
        x = np.linspace(-1,1,n)
        y = np.linspace(-1,1,n)

        [X,Y] = np.meshgrid(x,y)

        X_vec_temp = np.ravel(X)
        Y_vec_temp = np.ravel(Y)

        X_vec = X_vec_temp[np.logical_not(np.logical_and( X_vec_temp>0, Y_vec_temp <0))]
        Y_vec = Y_vec_temp[np.logical_not(np.logical_and( X_vec_temp>0, Y_vec_temp <0))]

        x = torch.from_numpy(X_vec).requires_grad_(requires_grad = True).type(torch.float32)
        y = torch.from_numpy(Y_vec).requires_grad_(requires_grad = True).type(torch.float32)

        x = torch.unsqueeze(x,-1)
        y = torch.unsqueeze(y,-1)
        
        Boundary = torch.logical_or(y == 1., x ==-1.)
        Boundary = torch.logical_or(Boundary, torch.logical_and(x<=0., y == -1.))
        Boundary = torch.logical_or(Boundary, torch.logical_and(x==0., y <=  0.))
        Boundary = torch.logical_or(Boundary, torch.logical_and(x>=0., y ==  0.))
        Boundary = torch.logical_or(Boundary, torch.logical_and(x==1., y >=  0.))
        
        Corners  = torch.logical_and(y == 1., x == -1.)
        Corners  = torch.logical_or(Corners, torch.logical_and(y == -1., x == -1.))
        Corners  = torch.logical_or(Corners, torch.logical_and(y ==  1., x ==  1.))
        Corners  = torch.logical_or(Corners, torch.logical_and(y == -1., x ==  0.))
        Corners  = torch.logical_or(Corners, torch.logical_and(y ==  0., x ==  0.))
        Corners  = torch.logical_or(Corners, torch.logical_and(y ==  0., x ==  1.))
        
        Reflex_corner = torch.logical_and(y ==  0., x ==  0.)

        return x, y, Boundary, Corners, Reflex_corner, 1/N

def Logical_Domains(N):
        n = 2*N+1
        x = np.linspace(-1,1,n)
        y = np.linspace(-1,1,n)

        [X,Y] = np.meshgrid(x,y)

        X_vec_temp = np.ravel(X)
        Y_vec_temp = np.ravel(Y)

        Bool = np.logical_not(np.logical_and( X_vec_temp >0, Y_vec_temp <0))
        return Bool, X, Y

def Int2D(f,Boundary, Corners, Reflex_corner, dx):         
        return (torch.sum(f) - 0.5*torch.sum(f[Boundary]) - 0.25*torch.sum(f[Corners]) + 0.5*torch.sum(f[Reflex_corner]))*(dx)**2

def Int1D(f,Boundary, dx):         
        return (torch.sum(f[Boundary]))*(dx)

if __name__ == "__main__":        
        npts = 3
        x, y, Boundary, Corners, dx = Domains(npts)
        print(x[Corners])
        print(y[Corners])
