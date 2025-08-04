import numpy as np
for i in [2, 4, 6, 8, 10]:
    data = np.loadtxt(f'EpUp_v_error_dim={i}.txt', delimiter = ',')
    #data = data[0::10,:]
    np.savetxt(f'EpUp_v_error_dim={i}.csv', data, delimiter = ',')

data = 0
for i in [1,2,3,4,5]:
    data += np.log(np.loadtxt(f'EpUp_v_error_dim=10_seed={i}.txt', delimiter = ' '))
data = np.exp(0.2*data)
np.savetxt(f'EpUp_v_error_dim=10.csv', data, delimiter = ',')
