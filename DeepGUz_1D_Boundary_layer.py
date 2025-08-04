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
# Parameter defaults
def para_def(epsilon = 1e-2, gamma = 1.0, EpUp = 20, NoUp = 200, h_n = 40, printyn = True, rho = 0.25*(1e-2)):
        return (epsilon, gamma, EpUp, NoUp, h_n, printyn, rho)
# Cartesian Grid 
def Domains(N, xStart = 0., xEnd = 1.):
        # 1-D Domains
        x1 = np.linspace(xStart, xEnd, N)
        x = torch.from_numpy(x1).requires_grad_(requires_grad = True).type(torch.float32)
        return torch.unsqueeze(x,-1)
# Trapezoidal Rule integral
def Int1D(fun):
        N = fun.size(dim=0)
        return torch.sum(0.5*(fun[0:N-1] + fun[1:N]))
##############################################################################################
###### Code for Plots and Saving ######
##############################################################################################
def plot_loss(loss_history, path, filename, para, labels = ['Total', 'PINNs', 'Adjoint', 'Gamma']):
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        for i in range(len(loss_history[0])):  # Loop over each component
                # Extract and plot each component
                component_loss = [loss[i] for loss in loss_history]
                plt.semilogy(list(range(len(component_loss))), component_loss, label=labels[i],  linewidth=0.5)
        def epochs2updates(x):
                return x / para[2]
        def updates2epochs(x):
                return x * para[2]
        secax = ax.secondary_xaxis('top', functions=(epochs2updates, updates2epochs))
        secax.set_xlabel('Update No.') 
        plt.legend()  
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
# Plot error        
def plot_error(para, err_his, path, filename):
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel('Update No.')
        ax.set_ylabel('Error')
        labels = ['State']
        for i in range(len(err_his[0])):  # Loop over each component
                # Extract and plot each component
                component_err = [err[i] for err in err_his]
                ax.semilogy(list(range(len(component_err))), component_err, label=labels[i],  linewidth=0.5)         
        def epochs2updates(x):
                return x * para[2]
        def updates2epochs(x):
                return x / para[2]
        secax = ax.secondary_xaxis('top', functions=(epochs2updates, updates2epochs))
        secax.set_xlabel('Epoch')
        plt.legend()
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
# My save
def mysave(f, path, filename, domain = True):
        if domain and torch.is_tensor(f):
                f = f.detach().numpy()
        elif torch.is_tensor(f):
                f = f.detach().numpy()
        np.savetxt(f"{path}/{filename}.csv", f, delimiter=',')
# Plot the solution and the exact solution
def plot_line(x, u, uexact, path,filename):
        plt.figure()
        plt.plot(x.detach().numpy(), uexact.detach().numpy(), '--', color='orange', label='$exact$', alpha=0.5)  
        plt.plot(x.detach().numpy(), u.detach().numpy(), 'o', color='orange', label='$u_\\theta$', alpha=0.5)  
        plt.grid(True)  # Turns on the grid
        plt.legend(loc='upper right')
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
# Plot the values of z
def plot_z(z, path,filename):
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel('Update No.') 
        labels = [r'$\lambda(0)$',r'$\lambda(1)$']
        for i in range(len(z[0])):  # Loop over each component
                # Extract and plot each component
                component_loss = [loss[i] for loss in z]
                plt.plot(list(range(len(component_loss))), component_loss, label=labels[i],  linewidth=0.5)
        def epochs2updates(x):
                return x * para[2]
        def updates2epochs(x):
                return x / para[2]
        secax = ax.secondary_xaxis('top', functions=(epochs2updates, updates2epochs))
        secax.set_xlabel('Epoch')
        plt.legend()  
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
##############################################################################################
###### Training of the Neural Network ######
##############################################################################################
def geo_train(f, gd, para, x, exact = torch.tensor([float('nan')]), learning_rate = 1e-3,  device = torch.device("cpu")):       
        ###### Activation Function ######
        class Swish(nn.Module):
                def __init__(self, inplace=True):
                        super(Swish, self).__init__()
                        self.inplace = inplace
                        
                def forward(self, x):
                        if self.inplace:
                                x.mul_(torch.sigmoid(x))
                                return x
                        else:
                                return x * torch.sigmoid(x)
        ###### Neural Network ######                
        class generatorNN(nn.Module):
                def __init__(self, input_n, h_n):
                        super(generatorNN, self).__init__()
                        self.log_learning_rate = nn.Parameter(torch.log(torch.tensor(1e-3)))
        
                        # Shared layers for both u and f
                        self.shared_layers = nn.Sequential(
                                nn.Linear(input_n,h_n),
                                #nn.ReLU(),
                                Swish(),
                                nn.Linear(h_n,h_n),
                                #nn.ReLU(),
                                Swish(),
                                nn.Linear(h_n,h_n),
                                #nn.ReLU(),
                                Swish(),
                        )
                        # Separate layers for u
                        self.u_layers = nn.Sequential(
                                nn.Linear(h_n, h_n),
                                Swish(),
                                nn.Linear(h_n, 1),
                        )
                        
                def forward(self, x):
                        shared_output = self.shared_layers(x)

                        # Compute u using its respective layers
                        u = self.u_layers(shared_output)
                        return u
        ###### Initialize the neural network using a standard method ######
        def init_normal(m):
                if type(m) == nn.Linear:
                        nn.init.kaiming_normal_(m.weight)
        ############################################################
        
        
        def generatorCriterion(x, z, f, gd, para):
                u = generator_NN(x)  # Unpacku 
                dx = ((torch.max(x) - torch.min(x))/ sqrt(x.size(0) - 1)).item()
                # Compute Laplacian
                ux  = torch.autograd.grad(u,  x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0] 
                # Convert target to a PyTorch tensor and ensure it's on the same device as x
                z = z.clone().detach()
                
                # First term: 0.5 * Norm{u - target}^2
                term1 = 0.5 * dx * para[0]* Int1D(ux**2) + 0.5 * dx * Int1D(u**2) - dx * Int1D(u*f)
                
                # Second term: alpha/2 * Norm{f}^2
                term2_1 = z[0]*(u[0]-gd[0]) + z[-1]*(u[-1]-gd[-1])
                #term2_2 = para[1] * ((para[0]*ux[0])**2 + (para[0]*ux[-1])**2)
                term2_2 = para[1]*0.5*(u[0]-gd[0])**2 + para[1]*0.5*(u[-1]-gd[-1])**2
                return term1 + term2_1 + term2_2, term1, term2_1, term2_2, u
        ############################################################
        dx = (torch.max(x) - torch.min(x))/ (x.size(0)- 1)
        z = torch.Tensor([0.0,0.0])
        z_history = []
        loss_history = []
        err_history = []
        input_n = 1
        ############################################################
        # use the modules apply function to recursively apply the initialization
        generator_NN = generatorNN(input_n, para[4]).to(device)
        generator_NN.apply(init_normal)
        generatorOptimiser = optim.Adam(generator_NN.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
        ############################################################
        for epoch in range(para[2]*para[3]):
                generator_NN.zero_grad()
                total_loss, PINNs, Adjoint, Gamma_term, u = generatorCriterion(x, z, f, gd, para)                   	
                total_loss.backward()
                generatorOptimiser.step()

                # Append the losses as a list
                
                if epoch % para[2] == 0:
                        loss_history.append([total_loss.item(), PINNs.item(), Adjoint.item(), Gamma_term.item()])
                        if para[5]:
                                print(f'Epoch: {epoch} \tTotal Loss: {total_loss.item()}, '
                                      f'PINNs loss: {PINNs.item()}, '
                                      f'Adjoint loss: {Adjoint.item()}, '
                                      f'Gamma_term loss: {Gamma_term.item()}')
                        state_error = torch.sqrt(Int1D(dx*(u-exact[0])**2))
                        err_history.append([state_error.item()])
                        if para[5]: print(f'State Error: ',err_history[-1][0])
                        z[0] = z[0] + para[6] * (u[0]-gd[0])
                        z[-1] = z[-1] + para[6] * (u[-1]-gd[-1])
                        z_history.append([z[0].item(), z[-1].item()])
                        # plot_line(x, u, exact[0], f'./Vary_epsilon_{log_eps}_EpUp_{para[2]}_NoUp_{para[3]}_nPts_{npts}_gamma_{log_g}_rho_{para[6].detach().item()}', f'Frame_epoch_{epoch}')
        u = generator_NN(x)
        state_error = torch.sqrt(Int1D(dx*(u-exact[0])**2))
        err_history.append([state_error.item()])
        return generator_NN, u, err_history, loss_history, z_history
##############################################################################################
###### Main Code ######
##############################################################################################
device = torch.device("cpu")
gamma_values  = [1E+0, 1E+1, 1E+2, 1E+3, 1E+4, 1E+5]
log_eps_values = [1, 3]
npts = 401
EpUp = 40
NoUp = 500
log_r_values = ['-inf']
for gamma in gamma_values:
        for log_eps in log_eps_values:
                eps = torch.Tensor([10**(-log_eps)])
                eps.to(torch.float64)
                for log_rho in log_r_values:
                        path = f'./Vary_epsilon_{log_eps}_gamma_{gamma}_rho_{log_rho}_EpUp_{EpUp}_NoUp_{NoUp}_nPts_{npts}'
                        #rho = torch.Tensor([10**(log_rho)])
                        rho = torch.Tensor([0])
                        rho.to(torch.float64)
                            
                        para = para_def(eps, gamma, EpUp, NoUp, 40, True, rho)
                        
                        
                        if not os.path.isdir(path):
                                os.makedirs(path)
                        x = Domains(npts)
                        f = torch.ones_like(x)  
                        gd = torch.Tensor([0.0,0.0])
                        ##############################################################################################
                        exact_u = 1 - (torch.exp((1-x.to(torch.float64))/torch.sqrt(eps).to(torch.float64)) + torch.exp(x.to(torch.float64)/torch.sqrt(eps).to(torch.float64)))/(torch.exp(1/torch.sqrt(eps).to(torch.float64)) + 1)
                        ##############################################################################################
                        generator_NN, u, err_history, loss_history, z_history = geo_train(f, gd, para, x, exact = [exact_u]) 
                        plot_line(x, u, exact_u, path, f'State')
                        plot_error(para, err_history, path, f'Error')
                        plot_z(z_history, path,f'Adjoint')
                        plot_loss(loss_history, path, f'Loss', para)
                        mysave(u, path, f'State')
                        mysave(u[0::4], path, f'State_small')
                        mysave(z_history, path, f'Adjoint')
                        mysave(z_history[0::5], path, f'Adjoint_small')
                        mysave(err_history, path, f'Error',False)
                        mysave(err_history[0::5], path, f'Error_small',False)
                        mysave(loss_history, path, f'Loss',False)
