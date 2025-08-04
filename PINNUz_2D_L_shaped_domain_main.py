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
# Misc
import L_shaped_preamble as LSP
import L_shaped_plots as plt
import os
##############################################################################################
###### Training of the Neural Network ######
##############################################################################################
def geo_train(x, y, Boundary, Corners, Reflex_corner, f, gD, u_exact, dx, epsilon = 0.01, rho = 1, gamma = 2, EpUp = 40, NoUp = 500, hn = 40, input_n = 1, device = torch.device("cpu")):
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
                        self.log_learning_rate = nn.Parameter(torch.log(torch.tensor(1e-4)))
        
                        # x-layers
                        self.x_layers = nn.Sequential(
                                nn.Linear(input_n,h_n),
                                Swish(),
                                nn.Linear(h_n,h_n),
                                Swish(),
                                nn.Linear(h_n,h_n),
                                Swish(),
                        )

                        # Energy layers
                        self.y_layers = nn.Sequential(
                                nn.Linear(input_n,h_n),
                                Swish(),
                                nn.Linear(h_n,h_n),
                                Swish(),
                                nn.Linear(h_n,h_n),
                                Swish(),
                        )
                        # Mixing layers
                        self.mix_layers_x_y = nn.Bilinear(h_n,h_n,h_n)

                        # Mixing layers
                        self.mix_layers_m_y = nn.Bilinear(h_n,h_n,h_n)

                        # Mixing layers
                        self.mix_layers_x_m = nn.Bilinear(h_n,h_n,h_n)
                        
                        # Separate layers for u
                        self.u_layers = nn.Sequential(
                                nn.Linear(h_n, h_n),
                                Swish(),
                                nn.Linear(h_n, h_n),
                                Swish(),
                                nn.Linear(h_n, input_n),
                        )
                        
                def forward(self, x, y):
                        x_trans = self.x_layers(x)
                        y_trans = self.y_layers(y)
                        mixed   = self.mix_layers_x_y(x_trans,y_trans)
                        #mixed   = self.mix_layers_m_y(mixed,y_trans)
                        #mixed   = self.mix_layers_x_m(x_trans,mixed)
                        u       = self.u_layers(mixed)
                        return u
                
        ###### Initialize the neural network using a standard method ######
        def init_normal(m):
                if type(m) == nn.Linear:
                        nn.init.kaiming_normal_(m.weight)
        ############################################################
        
        
        def generatorCriterion(x, y, f, gD, z, epsilon, gamma, Boundary, Corners, Reflex_corner, dx):
                # Unpack u
                u       = generator_NN(x,y)  

                # Compute Laplacian
                ux      = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                uxx     = torch.autograd.grad(ux,x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                uy      = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
                uyy     = torch.autograd.grad(uy,y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
                
                # Compute Residual
                R       = u - epsilon*(uxx+uyy)-f
                PINNs   = LSP.Int2D(R**2, Boundary, Corners, Reflex_corner, dx)

                # Gamma term
                term2   = 0.5*gamma*LSP.Int1D((u-gD)**2, Boundary, dx)

                # Detatch for torch reasons
                z       = z.clone().detach()

                # Adjoint term
                term3   = LSP.Int1D((u-gD)*z, Boundary, dx)

                return PINNs + term2 +term3, PINNs, term2, term3, u
        
        ############################################################
        ### Initialisation ###
        ############################################################
        
        # Pre-allocate tensors
        z = torch.zeros_like(x)                               # Adjiont function
        loss_history   = []
        adjoint_history= []
        error_history  = []                                                                 # number of input variables
        # Establish NN 
        generator_NN = generatorNN(input_n, hn).to(device)
        generator_NN.apply(init_normal)
        generatorOptimiser = optim.Adam(generator_NN.parameters(), lr=1e-4, betas = (0.9,0.99),eps = 10**-15)

        ############################################################
        ### Epoch Loop ###
        ############################################################

        for epoch in range(EpUp*NoUp+1):
                
                # Run the NN optimisation scheme
                generator_NN.zero_grad()
                total_loss, PINNS_loss, Gamma_loss, Adjoint_loss, u = generatorCriterion(x, y, f, gD, z, epsilon, gamma, Boundary, Corners, Reflex_corner, dx)
                total_loss.backward()
                generatorOptimiser.step()

                #for g in generatorOptimiser.param_groups:
                #        g['lr'] = 0.99975*(g['lr'] - 1E-4) + 1E-4
                

                ############################################################
                ### Update Step
                ############################################################
                if epoch % EpUp == 0:
                        # Update z values
                        loss_history.append([total_loss.item(), PINNS_loss.item(), Gamma_loss.item(), Adjoint_loss.item()])
                        z += rho*(u-gD)
                        adjoint_history.append( sqrt( (LSP.Int1D(z**2, Boundary, dx)).detach().item() ) )
                        error_history.append( sqrt( (LSP.Int2D((u-u_exact)**2, Boundary, Corners, Reflex_corner, dx)).detach().item() ) )
                        print(f'Epoch: {epoch} \tTotal Loss: {total_loss.item()} \tPINNs Loss: {PINNS_loss.item()} \tGamma Loss: {Gamma_loss.item()} \tAdjoint Loss: {Adjoint_loss.item()}')
        return u, z, loss_history, adjoint_history, error_history
##############################################################################################
###### Main Code ######
##############################################################################################
device  = torch.device("cpu")
npts    = 20
x, y, Boundary, Corners, Reflex_corner, dx     = LSP.Domains(npts)
r_values                        = [1E-4,1E-2,1,4,5]
log_eps_values                  = [1, 3]
gamma                           = 2
NoUp                            = 500
EpUp                            = 40
##############################################################################################
r = torch.sqrt(x**2 + y**2)
t = pi-torch.atan2( y, -x )
r1 = torch.sqrt((x+1)**2 + (y-1)**2)
t1 = torch.atan2( y-1 , x+1 )
u_exact = -(r**(2/3))*torch.sin(2*t/3)*(r1**(4/3))*torch.sin(2*t1)
# Compute Laplacian
u_exactx      = torch.autograd.grad(u_exact, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
u_exactxx     = torch.autograd.grad(u_exactx,x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
u_exacty      = torch.autograd.grad(u_exact, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
u_exactyy     = torch.autograd.grad(u_exacty,y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
                
# Compute Residual


################################################################################################
gD      = u_exact.detach()
################################################################################################
for log_eps in log_eps_values:
        epsilon = torch.Tensor([10**(-log_eps)])
        epsilon.to(torch.float64)
        f             = u_exact - epsilon*(u_exactxx+u_exactyy)
        corner0 = torch.nansum(f[r < 2*dx])/3
        corner1 = torch.nansum(f[r1 < 2*dx])/3
        f[torch.isnan(f)] = torch.Tensor([corner0, corner1])
        f = f.detach()
        for rho in r_values:
                path = f'./Vary_epsilon_{log_eps}_gamma_{gamma}_rho_{rho}_EpUp_{EpUp}_NoUp_{NoUp}_nPts_{npts}'

                if os.path.isdir(path):
                        continue
                else:
                        os.makedirs(path)
                u, z, loss_history, adjoint_history, error_history = geo_train(x, y, Boundary, Corners, Reflex_corner, f, gD, u_exact, dx, epsilon, rho, gamma)

                plt.plot_error(error_history, path, f'Error')
                plt.colour_plot(u.detach().numpy(), npts, path, 'State', title = '')
                
                plt.mysave(adjoint_history, path, f'Adjoint')
                plt.mysave(adjoint_history[0::5], path, f'Adjoint_small')
                plt.mysave(error_history, path, f'Error')
                plt.mysave(error_history[0::5], path, f'Error_small')
                plt.mysave(loss_history, path, f'Loss')
                plt.mysave(loss_history[0::5], path, f'Loss_small')
                
                plt.mysave(u, path, f'Table_u')
                plt.mysave(x, path, f'Table_x')
                plt.mysave(y, path, f'Table_y')
                plt.mysave(z, path, f'Table_z')
                
