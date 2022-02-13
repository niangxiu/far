# expressions of the dynamical system: the neural network from B. Cessac and J.A. Sepulchre, I changed the time-dependent perturbation and objective function to a location-dependent one. Soon FAR will handle time-dependent perturbations.
from __future__ import division
import numpy as np
from numpy import newaxis, sinh, cosh, tanh
import shutil
import sys
import itertools
from pdb import set_trace


# default settings
nstep = 20 # step per segment
nus = 1 # u in paper, number of homogeneous tangent solutions
nc = 9 # M in papaer, dimension of phase space

nprm = 3 # number of parameters
nseg_ps = 100
nseg_dis = 100 # segments to discard, not even for Javg
A = 1 # first parameter
g = 3 # second parameter


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
# J[8,8] is the third parameter


def fphi(x):
    f = A * J @ tanh(g*x)
    phi = (x**2).sum()
    return f, phi


def phix(x):
    phix = 2 * x
    return phix


def fx(x):
    # the first axis labels components of f
    fx = A * J * g / (cosh(g * x) ** 2)
    return fx


def fxx(x):
    # the first axis labels components of f
    fxx = np.zeros([nc,nc,nc])
    for i in range(nc):
        for j in range(nc):
            fxx[i,j,j] = A * J[i,j] * g**2 * -2 * sinh(g*x[j]) / cosh(g*x[j])**3
    return fxx


def fgafgax(x):
    # quantities related to X. Note fga_n = X_{n+1}
    # the first axis with length nc labels components of f
    fga = np.zeros([nprm,nc])
    fgax = np.zeros([nprm,nc,nc])

    fga[0] = J @ tanh(g * x)
    fga[1] = A * J @ (x / cosh(g*x)**2)
    fga[2] = A * tanh ( g*x[8] ) 
    
    for i in range(nc):
        fgax[0] = J * g / (cosh(g * x) ** 2)
        for j in range(nc):
            fgax[1,i,j] = A * J[i,j] / cosh(g*x[j])**2 \
                    + A * J[i,j] * x[j] * -2 * g * sinh(g*x[j]) / cosh(g*x[j])**3

    fgax[2,8,8] = A * g / cosh(g*x[8])**2

    return fga, fgax
