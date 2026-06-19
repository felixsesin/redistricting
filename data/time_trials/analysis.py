import numpy as np
from pathlib import Path
import pickle


path1 = Path(__file__).parent / '001-113.pkl'
path2 = Path(__file__).parent / '114-250.pkl'

with open(path1, 'rb') as f:
    dataset = pickle.load(f)

#with open(path2, 'rb') as f:
#    to_add = pickle.load(f)

#    for key, val in to_add.items():
#        dataset[key] = val

linear = False
quadratic = True

up_to = []
t_tot = []

for trial, data in dataset.items():

    n = data['IV_up_to']
    t = data['DV_total']

    """
        'DV_ensemble',  'DV_vertical', 'DV_horizontal',
        'DV_diffusion', 'DV_eigenset', 'DV_total'
    """

    up_to.append(n)
    t_tot.append(t)


# PLOTTER

import matplotlib.pyplot as plt

xRange = np.linspace(1,max(up_to), 1000)
yRange = lambda x: x

xData = np.array(up_to)
yData = np.array(t_tot)

if linear:

    coeffs = np.polyfit(up_to, t_tot, deg=1)
    m,b = coeffs
    y_pred = np.polyval(coeffs, xData)

    ss_res = np.sum((yData - y_pred)**2)
    ss_tot = np.sum((yData - np.mean(yData))**2)

    print('m = ', float(m))
    print('b = ', float(b))
    print('r2= ', float(1 - ss_res / ss_tot))

    yRange = lambda x: m*x + b 

if quadratic:

    coeffs = np.polyfit(up_to, t_tot, deg=2)
    a,b,c = coeffs
    y_pred = np.polyval(coeffs, xData)

    ss_res = np.sum((yData - y_pred)**2)
    ss_tot = np.sum((yData - np.mean(yData))**2)

    print('a = ', float(a))
    print('b = ', float(b))
    print('c = ', float(c))
    print('r2= ', float(1 - ss_res / ss_tot))

    yRange = lambda xRange: a*xRange*xRange + b*xRange + c



plt.scatter(up_to, t_tot)
plt.plot(xRange, yRange(xRange))

plt.grid(True)
plt.show()






exit()

i = 20

trial = list(dataset.keys())[i]

data = dataset[trial]

print('trial: ', trial)
print('up_to: ', data['IV_up_to'])
print('ensemble: ', data['DV_ensemble'])
print('vertical: ', data['DV_vertical'])
print('horizontal: ', data['DV_horizontal'])
print('diffusion: ', data['DV_diffusion'])
print('eigenset: ', data['DV_eigenset'])
print('total: ', data['DV_total'])


"""
['parameters', 'routine', 'IV_up_to', 'DV_ensemble',
'DV_vertical', 'DV_horizontal', 'DV_diffusion',
'DV_eigenset', 'DV_total']
"""