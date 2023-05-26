# expressions of the dynamical system: modified solonoid map with M=11 and u=10, with two parameters
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
nstep = 20 # step per segment
nus = 1 # u in paper, number of homogeneous tangent solutions
nc = 1 # M in papaer, dimension of phase space

nprm = 2 # number of parameters
nseg_ps = 10
nseg_dis = 10 # segments to discard, not even for Javg
prm = np.array([0., 0.]) # the epsilon on Patrick's paper
A = pi


def fphi(x):
    if 0 <= x <= 0.5:
        f = 2*x + prm[1]*sin(A*x)/A
    else:
        f = 2 - 2*x + prm[1]*sin(A*x)/A
    f = np.array(f % 1)
    phi = x
    return f, phi


def phix(x):
    phix = 1
    return phix


def fx(x):
    # the first axis labels components of f
    if 0 <= x <= 0.5:
        fx = 2 + prm[1]*cos(A*x)
    else:
        fx = -2 + prm[1]*cos(A*x)
    fx = np.array([[fx]])
    return fx


def fxx(x):
    # the first axis labels components of f
    fxx = -prm[1]*A*sin(A*x)
    fxx = np.array([[[fxx]]])
    return fxx


def fgafgax(x):
    # quantities related to X. Note fga_n = X_{n+1}
    # the first axis of length nc labels components of f
    fga = np.zeros([nprm,nc])
    fga[1,0] = sin(A*x)/A

    fgax = np.zeros([nprm,nc,nc])
    fgax[1,0,0] = cos(A*x)
    return fga, fgax
