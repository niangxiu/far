# expressions of the dynamical system: the neural network from B. Cessac and J.A. Sepulchre, I changed the time-dependent perturbation and objective function to a location-dependent one. Soon FAR will handle time-dependent perturbations.
from __future__ import division
import numpy as np
from numpy import newaxis, sinh, cosh, tanh
import shutil
import sys
import itertools
from pdb import set_trace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')
plt.rc('axes', titlesize='xx-large')

g = 3.0
J = np.array([
    [ 0.   ,  0.   ,  0.213,  0.   ,  0.469,  0.   ,  0.   ,  0.69 , 0.318],
    [-1.131,  0.822,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.007, -0.301],
    [ 0.   , -0.234,  0.   ,  0.   , -0.51 , -0.283, -0.177,  0.   , 0.   ],
    [ 0.   ,  0.644,  0.   ,  0.   ,  0.033,  0.   ,  1.187,  0.722, 0.   ],
    [ 0.   ,  0.   ,  0.   ,  0.   ,  0.511, -0.579, -0.495,  0.269, 0.   ],
    [ 0.   ,  1.015,  0.   ,  0.   ,  0.   ,  0.   , -1.312,  0.684, -0.365],
    [ 0.   ,  0.   , -0.852, -0.342,  0.389,  0.   ,  0.   , -0.041, 0.   ],
    [ 0.   ,  0.416,  0.   ,  0.   , -0.084,  0.   ,  0.287,  0.208, 0.   ],
    [ 0.   ,  0.   ,  0.   , -0.649,  0.   ,  0.   ,  0.331,  0.14 , -1.023]])

N = 40000
x = np.zeros((N,9))
# x[0] += 0.00000001

for i in range (N-1):
    x[i+1] = J @ (tanh(g*x[i])) + 0.1*np.random.rand(9)

plt.figure(figsize=[6,6])
plt.plot(x[1000:,0], x[1000:,1], '.', markersize=1)
plt.xlabel('x0')
plt.ylabel('x1')
plt.tight_layout()
plt.savefig('orbit.png')
plt.close()

