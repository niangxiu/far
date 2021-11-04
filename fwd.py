from __future__ import division
import numpy as np
from numpy import newaxis
import itertools
from pdb import set_trace
from scipy.linalg import block_diag
from ds import *
from misc import nanarray


def primal_raw(x0, n):
    # return quantities not related to fx: x, phi
    x = nanarray([n+1, nc])
    phi = nanarray([n+1])
    x[0] = x0
    for i in range(n):
        x[i+1], phi[i] = fphi(x[i])
    _, phi[n] = fphi(x[n])
    return x, phi


def primal(x0, nseg, W):
    # compute psi and reshape the raw results. Note that the returned x[0,0] is not x0.
    x = nanarray([nseg, nstep+1, nc]) # only for debug 
    psi = np.zeros([nseg, nstep+1])

    x_, phi_ = primal_raw(x0, nseg*nstep + 2*W)
    phiavg = phi_.mean()
    phi_ = phi_ - phiavg
    for k in range(nseg):
        x[k]= x_[k*nstep+W: k*nstep+W+nstep+1]
        for w in range(2*W+1):
            psi[k] += phi_[k*nstep+w: k*nstep+w+nstep+1]
    return x, psi, phiavg


def preprocess():
    # return the initial condition 
    np.random.seed()
    x0 = np.random.rand(nc)
    x, _ = primal_raw(x0, nseg_ps*nstep)
    return x[-1]


def tan(x, e):
    # compute homogeneous tangent equations
    for i in range(nstep):
        e[i+1] = fx(x[i]) @ e[i]
    return 


def Q0():
    # generate initial conditions for tangent solutions
    w0 = np.random.rand(nc, nus)
    Q0, _ = np.linalg.qr(w0, 'reduced')
    return Q0


def renorm(e):
    # renromalize e
    Q, R = np.linalg.qr(e, 'reduced')
    return Q, R


def getLE(R):
    I = np.arange(R.shape[-1])
    _ = np.log2(np.abs(R[1:-1,I,I]))
    LE = _.mean(axis=0) / nstep
    LE = 2**LE
    return LE


def fgafgax_(x):
    fga = nanarray([nseg, nstep+1, nc])
    fgax = nanarray([nseg, nstep+1, nc, nc])
    for al in range(nseg):
        for n in range(nstep+1):
            fga[al, n], fgax[al, n] = fgafgax(x[al,n])
    return fga, fgax
