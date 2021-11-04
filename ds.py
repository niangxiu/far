# expressions of the dynamical system: modified solonoid map with M=11 and u=10
from __future__ import division
import numpy as np
from numpy import newaxis, sin, cos
import shutil
import sys
import itertools
from pdb import set_trace


nstep = 10 # step per segment
nus = 2 # u in paper, number of homogeneous tangent solutions
nc = 3 # M in papaer, dimension of phase space
nseg = 10
nseg_ps = 100
nseg_dis = 100 # segments to discard, not even for Javg
prm = 0.1 # the epsilon on Patrick's paper
W = 10
A = 5
ii = list(range(1,nc))


def fphi(x):
    f = np.zeros(nc)
    f[0] = 0.05*x[0] + 0.1*cos(A*x[ii]).sum() + prm
    f[ii] = (2*x[ii] + prm*(1+x[0]) * sin(2*x[ii])) % (2*np.pi)
    phi = x[0]**3 + 0.005 * ((x[ii] - np.pi)**2).sum()
    return f, phi


def phix(x):
    phix = np.zeros(x.shape)
    phix[0] = 3 * x[0]**2
    phix[ii] = 0.005 * 2 * (x[ii] - np.pi)
    return phix


def fx(x):
    # the first axis labels components of f
    fx = np.zeros([nc,nc])
    fx[0,0] = 0.05
    fx[0,ii] = -0.1*A*sin(A*x[ii])
    fx[ii,0] = prm * sin(2*x[ii])
    fx[ii,ii] = 2 + prm*(1+x[0])*2*cos(2*x[ii])
    return fx


def fxx(x):
    # the first axis labels components of f
    fxx = np.zeros([nc,nc,nc])
    fxx[0,ii,ii] = -0.1 * A**2 * cos(A*x[ii])
    fxx[ii,0,ii] = 2 * prm * cos(2*x[ii])
    fxx[ii,ii,0] = 2 * prm * cos(2*x[ii])
    fxx[ii,ii,ii]= -prm * (1+x[0]) * 4 * sin(2*x[ii])
    return fxx


def fgafgax(x):
    # quantities related to X. Note fga_n = X_{n+1}
    # the first axis labels components of f
    fga = np.zeros(nc)
    fga[0] = 1
    fga[ii] = (1+x[0]) * sin(2*x[ii])

    fgax = np.zeros([nc,nc])
    fgax[ii,0] = sin(2*x[ii])
    fgax[ii,ii] = (1+x[0]) * 2* cos(2*x[ii])
    return fga, fgax
    
