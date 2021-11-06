from __future__ import division
import numpy as np
from numpy import newaxis
import itertools
from pdb import set_trace
from scipy.linalg import block_diag
from ds import *
import ds
import fwd, adj
from misc import nanarray


def far(nseg, W):
    Cinv = nanarray([nseg, nus, nus])
    R = nanarray([nseg+1, nus, nus]) # R[k] at k \Delta T
    depsnup, depsnupt = nanarray([2, nseg, nus])
    b, bt = nanarray([2, nseg+1, nus])

    # nup and eps in final seg, only use step 0; will cut last seg
    nup, nupt = nanarray([2, nseg+1, nstep+1, nc])
    e, eps = nanarray([2, nseg+1, nstep+1, nc, nus]) 

    # preprocess
    x0 = fwd.preprocess()
    x, psi, phiavg = fwd.primal(x0, nseg, W) # notice that x0 is changed

    # tangent
    e[0,0] = fwd.Q0()
    R[0] = np.eye(nus)
    for k in range(nseg):
        fwd.tan(x[k], e[k])
        e[k+1,0], R[k+1] = fwd.renorm(e[k,-1])
    LE = fwd.getLE(R)

    # adjoint
    eps[nseg,0] = e[nseg,0] 
    nup[nseg,0], nupt[nseg,0] = 0, 0
    for k in range(nseg-1,-1,-1):
        eps[k,-1], nup[k,-1], nupt[k,-1], b[k+1], bt[k+1] \
                = adj.renorm(eps[k+1,0], nup[k+1,0], nupt[k+1,0], e[k,-1])
        Cinv[k], depsnup[k], depsnupt[k] \
                = adj.products(x[k], e[k], eps[k], nup[k], nupt[k])

    # cut the nans in e, eps, nup, nupt
    e = e[:-1]
    eps = eps[:-1]
    nup = nup[:-1]
    nupt = nupt[:-1]

    # adjoint shadowing
    aa = adj.nias(Cinv, depsnup, R, b)
    aat = adj.nias(Cinv, depsnupt, R, bt)
    nu = nup + (eps*aa[:,newaxis,newaxis,:]).sum(-1) 
    nut = nupt + (eps*aat[:,newaxis,newaxis,:]).sum(-1) 

    # finally, get X related quantities and compute the adjoint response
    sc_, uc_ = nanarray([2, nprm, nseg, nstep])
    for k in range(nseg):
        fga, fgax = fwd.fgafgax_(x[k,:-1])
        sc_[:,k] = (nu[k][newaxis,1:] * fga).sum(-1) # shadowing contribution
        div1 = (nut[k][newaxis,1:] * fga).sum(-1) # unstable divergence part 1
        div2 = (eps[k][newaxis,1:].transpose(0,1,3,2) @ fgax @ e[k][newaxis,:-1]).trace(axis1=-1, axis2=-2)
        uc_[:,k] = (psi[k][newaxis,1:] * (div1 + div2))
    sc = sc_.mean((-1,-2))
    uc = uc_.mean((-1,-2))
    
    return phiavg, sc, uc, x, nu, sc_, nut, LE
