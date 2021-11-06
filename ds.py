# expressions of the dynamical system: modified solonoid map with M=11 and u=10
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
nstep = 20 # step per segment
nus = 2 # u in paper, number of homogeneous tangent solutions
nc = 3 # M in papaer, dimension of phase space

nprm = 2 # number of parameters
nseg_ps = 10
nseg_dis = 10 # segments to discard, not even for Javg
prm = np.array([0.1, 0.1]) # the epsilon on Patrick's paper
A = 5
ii = list(range(1,nc))
ucweight = 0.02


def fphi(x):
    f = np.zeros(nc)
    f[0] = cr*x[0] + 0.1*cos(A*x[ii]).sum() + prm[0]
    f[ii] = (2*x[ii] + prm[1]*(1+x[0]) * sin(2*x[ii])) % (2*np.pi)
    phi = x[0]**3 + ucweight * ((x[ii] - np.pi)**2).sum()
    return f, phi


def phix(x):
    phix = np.zeros(x.shape)
    phix[0] = 3 * x[0]**2
    phix[ii] = ucweight * 2 * (x[ii] - np.pi)
    return phix


def fx(x):
    # the first axis labels components of f
    fx = np.zeros([nc,nc])
    fx[0,0] = cr
    fx[0,ii] = -0.1*A*sin(A*x[ii])
    fx[ii,0] = prm[1] * sin(2*x[ii])
    fx[ii,ii] = 2 + prm[1]*(1+x[0])*2*cos(2*x[ii])
    return fx


def fxx(x):
    # the first axis labels components of f
    fxx = np.zeros([nc,nc,nc])
    fxx[0,ii,ii] = -0.1 * A**2 * cos(A*x[ii])
    fxx[ii,0,ii] = 2 * prm[1] * cos(2*x[ii])
    fxx[ii,ii,0] = 2 * prm[1] * cos(2*x[ii])
    fxx[ii,ii,ii]= -prm[1] * (1+x[0]) * 4 * sin(2*x[ii])
    return fxx


def fgafgax(x):
    # quantities related to X. Note fga_n = X_{n+1}
    # the first axis labels components of f
    fga = np.zeros([nprm,nc])
    fga[0,0] = 1
    fga[1,ii] = (1+x[0]) * sin(2*x[ii])

    fgax = np.zeros([nprm,nc,nc])
    fgax[1,ii,0] = sin(2*x[ii])
    fgax[1,ii,ii] = (1+x[0]) * 2* cos(2*x[ii])
    return fga, fgax
    
