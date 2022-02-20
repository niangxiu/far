from __future__ import division
import numpy as np
from numpy import newaxis
import itertools
from pdb import set_trace
from scipy.linalg import block_diag
from numpy.linalg import solve
from ds import *
import ds
from misc import *


def adj(x, e, eps, nup, nupt):
    # eps, nup, nupt are changed inside this function
    for n in range(nstep, 0, -1):
        M = fx(x[n-1]).T
        eps[n-1] = M @ eps[n]
        nup[n-1] = M @ nup[n] + phix(x[n-1])
        omega = (eps[n][:,newaxis,newaxis,:] * fxx(x[n-1])[:,:,:,newaxis]\
                * e[n-1][newaxis,:,newaxis,:]).sum((0,1,3))
        nupt[n-1] = M @ nupt[n] + omega
    return


def renorm(eps0, nup0, nupt0, eN):
    # renormalization of adjoint solutions across interfaces of segments
    epsN = np.linalg.solve(eps0.T @ eN, eps0.T).T
    M = eps0.T @ eps0
    b = np.linalg.solve(M, eps0.T @ nup0)
    nupN = nup0 - eps0 @ b
    bt = np.linalg.solve(M, eps0.T @ nupt0)
    nuptN = nupt0 - eps0 @ bt
    return epsN, nupN, nuptN, b, bt


def nias(R, b):
    # solve the nonintrusive adjoint shadowing problem
    nseg = R.shape[0] - 1
    a = nanarray([nseg, nus])

    a[0] = -b[0]
    for i in range(1, nseg):
        a[i] = solve(R[i].T, a[i-1]) - b[i]
   
    return a
