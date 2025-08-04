##############################################################################################
###### Packages to Import ######
##############################################################################################
import numpy as np
from math import pi, sqrt, exp
import csv
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
import L_shaped_preamble as Lsp
import torch

##############################################################################################
###### Code for Plots and Saving ######
##############################################################################################
# My save
def mysave(f, path, filename):
        if torch.is_tensor(f):
                f = f.detach().numpy()
        np.savetxt(f"{path}/{filename}.csv", f, delimiter=',')
        
##############################################################################################
###### Plots ######
##############################################################################################

def plot_loss(loss_history, path, filename, labels = ['Total', 'PINNs', r'$\gamma$-term', 'Adjoint']):
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel('Update No.')
        ax.set_ylabel('Loss')
        for i in range(len(loss_history[0])):  # Loop over each component
                # Extract and plot each component
                component_loss = [loss[i] for loss in loss_history]
                plt.semilogy(list(range(len(component_loss))), component_loss, label=labels[i],  linewidth=0.5)
        plt.legend()  
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()

def colour_plot(f, N, path,filename, title = '', cont_max = 30):
        Bool, X, Y = Lsp.Logical_Domains(N)
        F_vec = np.ravel(np.nan*np.ones_like(X))
        F_vec[Bool] = f.reshape((-1,))
        F = F_vec.reshape((2*N+1,2*N+1))
        plt.contourf(X,Y,F, cont_max, cmap = 'plasma') 
        plt.colorbar()
        plt.contour(X,Y,F, cont_max, colors=['#FEFBEA'], linewidths = [0.25])
        plt.title(title)
        plt.fill("j", "k", 'w', 
         data={"j": [ 0, 0, 1,  1], 
               "k": [-1, 0, 0, -1]})
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
        
def plot_error(err_his, path, filename):
        plt.semilogy(list(range(len(err_his))), err_his, linewidth=0.5)
        plt.xlabel('Update No.')
        plt.ylabel('Error')
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
        
if __name__ == "__main__":        
        npts = 101
        x, y, Boundary, Corners, Reflect_corners, dx = Lsp.Domains(npts)
        r = np.sqrt((x.detach().numpy())**2 + (y.detach().numpy())**2)
        t = pi-np.atan2( y.detach().numpy() , -x.detach().numpy() )

        r1 = np.sqrt((x.detach().numpy()+1)**2 + (y.detach().numpy()-1)**2)
        t1 = np.atan2( y.detach().numpy()-1 , x.detach().numpy()+1 )
        
        f = -(r**(2/3))*np.sin(2*t/3)*(r1**(2/3))*np.sin(2*t1)
        colour_plot(f, npts, '.', 'state', title = '')
