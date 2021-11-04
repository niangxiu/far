from __future__ import division
import numpy as np
from numpy import newaxis
import itertools
from pdb import set_trace
from scipy.linalg import block_diag
from ds import *
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


def products(x, e, eps, nup, nupt):
    # return inner products on one segment
    adj(x, e, eps, nup, nupt)
    C = (eps[:,:,:,newaxis] * eps[:,:,newaxis,:])[1:].sum((0,1))
    Cinv = np.linalg.inv(C)
    depsnup = (eps * nup[:,:,newaxis])[1:].sum((0,1))
    depsnupt = (eps * nupt[:,:,newaxis])[1:].sum((0,1))
    return Cinv, depsnup, depsnupt


def renorm(eps0, nup0, nupt0, eN):
    # renormalization of adjoint solutions across interfaces of segments
    epsN = np.linalg.solve(eps0.T @ eN, eps0.T).T
    M = eps0.T @ eps0
    b = np.linalg.solve(M, eps0.T @ nup0)
    nupN = nup0 - eps0 @ b
    bt = np.linalg.solve(M, eps0.T @ nupt0)
    nuptN = nupt0 - eps0 @ bt
    return epsN, nupN, nuptN, b, bt


def nias(Cinv, d, R, b):
    # solve the nonintrusive adjoint shadowing problem
    D, E = nanarray([2, nseg, nus, nus])
    y, lbd, a = nanarray([3, nseg, nus])

    for i in range(1, nseg):
        D[i] = Cinv[i] @ R[i]
        E[i] = Cinv[i-1] + R[i].T @ D[i] 
        y[i] = D[i].T @ d[i] - Cinv[i-1] @ d[i-1] - R[i].T @ b[i]

    # forward chasing. Use the new E!
    for i in range(2, nseg):
        W = np.linalg.solve(E[i-1].T, D[i-1].T).T 
        E[i] -= W @ D[i-1].T
        y[i] += W @ y[i-1]

    # backward chasing
    lbd[nseg-1] = np.linalg.solve(E[nseg-1], y[nseg-1]) 
    for i in range(nseg-2, 0, -1):
        lbd[i] = np.linalg.solve(E[i], (D[i].T @ lbd[i+1] + y[i]))

    # compute a from lbd
    a[0] = Cinv[0] @ (-d[0] - lbd[1])
    for i in range(1, nseg-1):
        a[i] = Cinv[i] @ (-d[i] - lbd[i+1] + R[i]@lbd[i])
    a[nseg-1] = Cinv[nseg-1] @ (-d[nseg-1] + R[nseg-1]@lbd[nseg-1])

    return a


# def tan2nd(rini, u, psi, w, vt):
    # r = rini
    # for i in range(nstep):
        # fu, _ = fufs(u[i])
        # fuu, fsu = fuufsu(u[i])
        # rn = fu @ r + psi[i+1] * fsu @ w[i] + fuu @ vt[i] @ w[i]
        # r = rn
    # return r
