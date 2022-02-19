# expressions of the dynamical system: modified solonoid map with M=11 and u=10, with two parameters
from __future__ import division
import numpy as np
from numpy import newaxis, sin, cos
import shutil
import sys
import itertools
from pdb import set_trace

# to use nc = 3 for robustness
# nstep = 5 # step per segment
# nus = nc = 3 # u in paper, number of homogeneous tangent solutions
# cr = 0.2 # contracting rate of the first variable

# default settings
cr = 0.05 # contracting rate of the first variable
nstep = 10 # step per segment
nus = 2 # u in paper, number of homogeneous tangent solutions
nc = 2 # M in papaer, dimension of phase space

nprm = 2 # number of parameters
nseg_ps = 10
nseg_dis = 10 # segments to discard, not even for Javg
prm = np.array([0.1, 0.1]) # the epsilon on Patrick's paper
A = 5
ii = list(range(nc))


def fphi(x):
    f = (2*x + prm[1]*sin(2*x) + 10*np.random.rand(2)) % (2*np.pi)
    phi = ((x - np.pi)**2).sum()
    return f, phi


def phix(x):
    phix = 2 * (x - np.pi)
    return phix


def fx(x):
    # the first axis labels components of f
    fx = np.zeros([nc, nc])
    fx[ii,ii] = 2 + prm[1]*2*cos(2*x[ii])
    return fx


def fxx(x):
    # the first axis labels components of f
    fxx = np.zeros([nc,nc,nc])
    fxx[ii,ii,ii]= -prm[1] * 4 * sin(2*x[ii])
    return fxx


def fgafgax(x):
    # quantities related to X. Note fga_n = X_{n+1}
    # the first axis of length nc labels components of f
    fga = np.zeros([nprm,nc])
    fga[1,ii] = sin(2*x[ii])

    fgax = np.zeros([nprm,nc,nc])
    fgax[1,ii,ii] = 2 * cos(2*x[ii])
    return fga, fgax
    
