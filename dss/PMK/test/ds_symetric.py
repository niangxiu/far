# expressions of the dynamical system: from Alexey Korepanov 2016 Nonlinearity paper
from __future__ import division
import numpy as np
from numpy import newaxis, sin, cos, pi
import shutil
import sys
import itertools
from pdb import set_trace

# to use nc = 3 for robustness
# nstep = 5 # step per segment
# nus = nc = 3 # u in paper, number of homogeneous tangent solutions
# cr = 0.2 # contracting rate of the first variable

# default settings
nstep = 40 # step per segment
nus = 1 # u in paper, number of homogeneous tangent solutions
nc = 1 # M in papaer, dimension of phase space

nprm = 2 # number of parameters
nseg_ps = 10
nseg_dis = 10 # segments to discard, not even for Javg
prm = np.array([0., 0.]) # the epsilon on Patrick's paper


def fphi(x):
    def ff(x):
        return x*(1+(2*x)**prm[1])
    if 0 <= x <= 0.5:
        f = ff(x)
    else:
        f = 1 - ff(1-x)
    f = np.array(f % 1)
    phi = cos(2*pi*x)
    # phi = 1
    return f, phi


def phix(x):
    phix = -2*pi* sin(2*pi*x)
    # phix = 0
    return phix


def fx(x):
    # the first axis labels components of f
    def ffx(x):
        return 1 + (1+prm[1])*(2*x)**prm[1] 
    if 0 <= x <= 0.5:
        fx = ffx(x)
    else:
        fx = ffx(1-x)
    fx = np.array([[fx]])
    return fx


def fxx(x):
    # the first axis labels components of f
    def ffxx(x):
        return (1+prm[1])*2*prm[1]*(2*x)**(prm[1]-1) 
    if 0<=x<=0.5:
        fxx = ffxx(x)
    else:
        fxx = - ffxx(1-x)
    fxx = np.array([[[fxx]]])
    return fxx


def fgafgax(x):
    # quantities related to X. Note fga_n = X_{n+1}
    # the first axis of length nc labels components of f
    fga = np.zeros([nprm,nc])
    def ffga(x):
        return x*(2*x)**prm[1]*np.log(2*x)
    if 0<=x<=0.5:
        fga[1,0] = ffga(x)
    else:
        fga[1,0] = -ffga(1-x)

    fgax = np.zeros([nprm,nc,nc])
    def ffgax(x):
        return (2*x)**prm[1] + (1+prm[1])*(2*x)**prm[1]*np.log(2*x)
    if 0<=x<=0.5:
        fgax[1,0,0] = ffgax(x)
    else:
        fgax[1,0,0] = ffgax(1-x)
    return fga, fgax
