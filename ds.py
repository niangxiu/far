# expressions of the dynamical system: the neural network from B. Cessac and J.A. Sepulchre, I changed the time-dependent perturbation and objective function to a location-dependent one. Soon FAR will handle time-dependent perturbations.
from __future__ import division
import numpy as np
from numpy import newaxis, sin, cos, sinh, cosh, tanh
import shutil
import sys
import itertools
from pdb import set_trace


# default settings
nstep = 20 # step per segment
nus = 1 # u in paper, number of homogeneous tangent solutions
nc = 9 # M in papaer, dimension of phase space

nprm = 3 # number of parameters
nseg_ps = 10
nseg_dis = 10 # segments to discard, not even for Javg
prm = np.array([0.1, 0.1]) 
g = 3 


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


def fphi(x):
    f = J @ tanh(x)
    phi = x**2.sum()
    return f, phi


def phix(x):
    phix = 2 * x
    return phix


def fx(x):
    # the first axis labels components of f
    _ = g / (cosh(g * x) ** 2)
    fx = J * _
    return fx


def fxx(x):
    # the first axis labels components of f
    fxx = np.zeros([nc,nc,nc])
    for i in range(nc):
        for j in range(nc):
            fxx[i,j,j] = J[i,j] * g**2 * -2 * sinh(g*x[j]) / cosh(g*x[j])**3
    return fxx


def fgafgax(x):
    # quantities related to X. Note fga_n = X_{n+1}
    # the first axis labels components of f
    fga = np.zeros([nprm,nc])
    fgax = np.zeros([nprm,nc,nc])

    f = f(x)
    fx = fx(x)
    fga[0] = sin(f)
    fga[1] = J @ (x / cosh(g*x)**2)
    fga[2] = tanh ( g*x[8] ) 
    
    for i in range(nc):
        for j in range(nc):
            fgax[0,i,j] = cos(f[i]) * fx[i,j]
            fgax[1,i,j] = J[i,j] / cosh(g*x[j])**2 \\
                    + J[i,j] * x[j] * -2 * g * sinh(g*x[j]) / cosh(g*x[j]) ** 3

    fgax[2,8,8] = g / cosh(g*x[8]) ** 2

    return fga, fgax
