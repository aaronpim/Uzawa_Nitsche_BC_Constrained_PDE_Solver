import numpy as np
import matplotlib.pyplot as plt

max_seed = 3

pen_fin = []
uzw_fin = []
hard_fin = []

for Dim in range(2,11):

    pen = 0
    uzw = 0
    hard = 0
    for seed in range(1,max_seed+1):
        file_name = f"Error_Poisson_Penalty_dim={Dim}_Penalty=5_SEED={seed}.txt"
        pen += np.log(np.loadtxt(file_name))

        file_name = f"Error_Poisson_Uzawa_dim={Dim}_Penalty=5_SEED={seed}_rho=0.1.txt"
        uzw += np.log(np.loadtxt(file_name))

        file_name = f"Error_Poisson_Hard_dim={Dim}_SEED={seed}.txt"
        hard += np.log(np.loadtxt(file_name))

    pen = np.exp(pen/max_seed)
    uzw = np.exp(uzw/max_seed)
    hard = np.exp(hard/max_seed)

    np.savetxt(f'Penalty_dim={Dim}.csv', pen, delimiter=",")
    np.savetxt(f'Uzawa_dim={Dim}.csv', uzw, delimiter=",")
    np.savetxt(f'Hard_dim={Dim}.csv', hard, delimiter=",")

    pen_fin.append(np.mean(pen[-10:-1,1]))
    uzw_fin.append(np.mean(uzw[-10:-1,1]))
    hard_fin.append(np.mean(hard[-10:-1,1]))

    plt.figure()
    plt.semilogy(pen[:,0], pen[:,1], label = f'Penalty')
    plt.semilogy(uzw[:,0], uzw[:,1], label = f'Uzawa')
    plt.semilogy(hard[:,0], hard[:,1], label = f'Hard')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('L2-error')
    plt.savefig(f'Error_Poisson_{Dim}dim.pdf')
    plt.close()

np.savetxt(f'Penalty_final.csv', pen_fin)
np.savetxt(f'Uzawa_final.csv', uzw_fin)
np.savetxt(f'Hard_final.csv', hard_fin)
