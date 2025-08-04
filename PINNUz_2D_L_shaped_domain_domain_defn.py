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

# Generates a 2D Cartesian grid and classifies domain regions
def Domains(N):
        """
        Creates a 2D grid over the domain [-1, 1] x [-1, 1], excluding the bottom-right quadrant.
        Identifies boundary points, corners, and a reflex corner.
        
        Args:
            N (int): Half the number of intervals along each axis. Total grid size is (2N+1)^2.
        
        Returns:
            x (torch.Tensor): x-coordinates of valid domain points (column vector).
            y (torch.Tensor): y-coordinates of valid domain points (column vector).
            Boundary (torch.BoolTensor): Boolean mask indicating boundary points.
            Corners (torch.BoolTensor): Boolean mask indicating corner points.
            Reflex_corner (torch.BoolTensor): Boolean mask for a special corner (origin).
            dx (float): Grid spacing (1/N).
        """
        n = 2 * N + 1                           # Number of grid points along one axis
        x = np.linspace(-1, 1, n)              # 1D grid along x-axis
        y = np.linspace(-1, 1, n)              # 1D grid along y-axis

        [X, Y] = np.meshgrid(x, y)             # Create 2D meshgrid
        X_vec_temp = np.ravel(X)              # Flatten the X grid
        Y_vec_temp = np.ravel(Y)              # Flatten the Y grid

        # Remove points from bottom-right quadrant (x > 0 and y < 0)
        mask = np.logical_not(np.logical_and(X_vec_temp > 0, Y_vec_temp < 0))
        X_vec = X_vec_temp[mask]
        Y_vec = Y_vec_temp[mask]

        # Convert to PyTorch tensors with gradient tracking
        x = torch.from_numpy(X_vec).requires_grad_(True).type(torch.float32).unsqueeze(-1)
        y = torch.from_numpy(Y_vec).requires_grad_(True).type(torch.float32).unsqueeze(-1)

        # Identify boundary points (edges of the domain)
        Boundary = torch.logical_or(y == 1., x == -1.)
        Boundary = torch.logical_or(Boundary, torch.logical_and(x <= 0., y == -1.))
        Boundary = torch.logical_or(Boundary, torch.logical_and(x == 0., y <=  0.))
        Boundary = torch.logical_or(Boundary, torch.logical_and(x >= 0., y ==  0.))
        Boundary = torch.logical_or(Boundary, torch.logical_and(x == 1., y >=  0.))

        # Identify corner points
        Corners  = torch.logical_and(y == 1., x == -1.)
        Corners  = torch.logical_or(Corners, torch.logical_and(y == -1., x == -1.))
        Corners  = torch.logical_or(Corners, torch.logical_and(y ==  1., x ==  1.))
        Corners  = torch.logical_or(Corners, torch.logical_and(y == -1., x ==  0.))
        Corners  = torch.logical_or(Corners, torch.logical_and(y ==  0., x ==  0.))
        Corners  = torch.logical_or(Corners, torch.logical_and(y ==  0., x ==  1.))

        # Identify reflex corner (shared corner at origin)
        Reflex_corner = torch.logical_and(y == 0., x == 0.)

        return x, y, Boundary, Corners, Reflex_corner, 1/N


# Logical grid definition (returns Boolean mask and 2D meshgrid)
def Logical_Domains(N):
        """
        Creates a meshgrid and a Boolean mask to identify valid domain points,
        excluding the bottom-right quadrant.
        
        Args:
            N (int): Grid refinement parameter.
        
        Returns:
            Bool (np.ndarray): Boolean mask indicating valid domain points.
            X, Y (np.ndarray): Full meshgrid arrays (including excluded points).
        """
        n = 2 * N + 1
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        [X, Y] = np.meshgrid(x, y)

        X_vec_temp = np.ravel(X)
        Y_vec_temp = np.ravel(Y)

        Bool = np.logical_not(np.logical_and(X_vec_temp > 0, Y_vec_temp < 0))  # Remove bottom-right
        return Bool, X, Y


# 2D numerical integration over the domain with correction for boundaries and corners
def Int2D(f, Boundary, Corners, Reflex_corner, dx):
        """
        Performs 2D numerical integration over a structured grid using the midpoint rule,
        with adjustments at boundaries and corners to maintain accuracy.
        
        Args:
            f (torch.Tensor): Scalar field defined on the domain.
            Boundary (BoolTensor): Mask for boundary points.
            Corners (BoolTensor): Mask for corners.
            Reflex_corner (BoolTensor): Mask for shared corner point.
            dx (float): Grid spacing.
        
        Returns:
            torch.Tensor: Approximated integral over the domain.
        """
        return (torch.sum(f) 
                - 0.5 * torch.sum(f[Boundary]) 
                - 0.25 * torch.sum(f[Corners]) 
                + 0.5 * torch.sum(f[Reflex_corner])) * dx**2


# 1D numerical integration over boundary
def Int1D(f, Boundary, dx):
        """
        Performs 1D numerical integration over the boundary points.
        
        Args:
            f (torch.Tensor): Scalar field defined on the domain.
            Boundary (BoolTensor): Mask for boundary points.
            dx (float): Grid spacing.
        
        Returns:
            torch.Tensor: Approximated integral along the boundary.
        """
        return torch.sum(f[Boundary]) * dx


# Script block for testing the domain logic
if __name__ == "__main__":
        npts = 3                              # Grid resolution parameter
        x, y, Boundary, Corners, dx = Domains(npts)[:5]  # Only use needed values
        print(x[Corners])                     # Print x-coordinates of corners
        print(y[Corners])                     # Print y-coordinates of corners
