# for various plots, including calls to main functions
from __future__ import division
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import shutil
import sys
import os
import time
import pickle
import itertools
from multiprocessing import Pool, current_process
from pdb import set_trace
from ds import *
import ds
from far import far
from fwd import primal_raw, preprocess
from misc import *

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# default parameters that changes frequently
nseg = 1000
W = 15
n_repeat = 4
ncpu = 2


def pr(nTstep):
    return primal_raw(preprocess(), nTstep)


def wrapped_primal(prm, nseg, n_repeat): 
    ds.prm = prm
    nTstep = nseg * ds.nstep
    arguments = [(nTstep,) for i in range(n_repeat)]
    if n_repeat == 1:
        results = [pr(*arguments[0])]
    else:
        with Pool(processes=ncpu) as pool:
            results = pool.starmap(pr, arguments)
    _, phi = zip(*results)
    phiavgs = np.array(phi).mean(axis = -1)
    print('prm, nseg, phiavg')
    [print('{}, {}, {}'.format(ds.prm, nseg, phiavg)) for phiavg in phiavgs]
    return phiavgs


def wrapped_far(prm, nseg, W, n_repeat): 
    ds.prm = prm
    arguments = [(nseg, W,) for i in range(n_repeat)]
    if n_repeat == 1:
        results = [far(*arguments[0])]
    else:
        with Pool(processes=ncpu) as pool:
            results = pool.starmap(far, arguments)
    phiavg_, sc_, uc_, *_ = zip(*results)
    print('prm, nseg, W, phiavg, sc, uc, grad')
    [print('{}, {:d}, {:d}, {:.2e}, {}, {}, {}'\
            .format(ds.prm, nseg, W, phiavg, sc, uc, sc-uc)) \
            for phiavg, sc, uc in zip(phiavg_, sc_, uc_)]
    return np.array(phiavg_), np.array(sc_), np.array(uc_)


def graph():
    x = np.linspace(0,1,1001)
    fphi = np.vectorize(ds.fphi)
    fig = plt.figure(figsize=(6,6))

    f, _ = fphi(x)
    plt.plot(x, f, 'k-', markersize=0.5, label="$\gamma$=0")
    ds.prm[1] = 1
    ds.A = 2 * np.pi * 1
    f, _ = fphi(x)
    plt.plot(x, f, '--', markersize=0.5, label="$\gamma$=1, T=1")
    ds.prm[1] = 8
    ds.A = 2 * np.pi * 8
    f, _ = fphi(x)
    plt.plot(x, f, '-.', markersize=0.5, label="$\gamma$=8, T=8")

    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.05, 1.15)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.tight_layout()
    plt.savefig('graph_B{}.png'.format(ds.B))
    plt.close()


def trajectory():
    ds.prm[1] = 0
    np.random.seed()
    starttime = time.time()
    x, phi = primal_raw(preprocess(), 200*10000)
    endtime = time.time()
    print('time for compute trajectory:', endtime-starttime)
    x = x[1000:]
    plt.figure(figsize=[6,6])
    # plt.plot(x.reshape(-1), np.random.rand(x.reshape(-1).shape[-1]), '.', markersize=0.3)
    plt.hist(x,20,density=True,histtype='step')
    plt.xlabel('$x$')
    plt.ylabel('$\\rho (x)$')
    plt.tight_layout()
    plt.savefig('x_frequent_B{}_ga{}.png'.format(ds.B, ds.prm[1]))
    plt.close()


def all_info():
    # generate all info prm = prm[0] = pr[1]
    starttime = time.time()
    phiavg, sc, uc, x, nu, sc_, nut, LE = far(200, W)
    endtime = time.time()
    print('time for far:', endtime-starttime)
    print('phiavg, sc, uc, grad = ', '{}, {}, {}, {}'.format(phiavg, sc, uc, sc-uc))
    print('Lyapunov exponenets = ', LE)
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13,12))
    nun = np.linalg.norm(nu, axis=-1).reshape(-1)
    hax = np.arange(nun.shape[0])
    ax1.plot(hax,nun)
    ax1.set_title('nu norm')
    sc_ = sc_.sum(0).reshape(-1)
    ax2.plot(np.arange(sc_.shape[0]),sc_)
    ax2.set_title('sc at each step')
    ax3.scatter(x[:,:,0], nu[:,:,0])
    ax3.set_title('x[0] vs nu[0]')
    nutn = np.linalg.norm(nut, axis=-1).reshape(-1)
    ax4.plot(hax,nutn)
    ax4.set_title('nutilde norm')
    plt.tight_layout()
    plt.savefig('all info.png')
    plt.close()


def change_prm():
    # grad for different prm = prm[0] = pr[1]
    n_repeat = 1 # must use 1, since prm in ds.py is fixed when the pool intializes
    A = 0.015 # step size in the plot
    NN = 12 # number of steps in parameters
    prms = np.tile(np.linspace(-0.3, 0.3, NN), (nprm,1)).T
    phiavgs = nanarray(NN)
    sc = nanarray(prms.shape)
    uc = nanarray(prms.shape)
    try:
        prms, phiavgs, grads = pickle.load(open("change_prm_B{}_T{}.p".format(ds.B,ds.T), "rb"))
    except FileNotFoundError:
        for i, prm in enumerate(prms):
            _, sc[i], uc[i] = wrapped_far(prm, 1000, W, n_repeat)
            phiavgs[i] = wrapped_primal(prm, 100000, n_repeat)
        grads = (sc - uc).sum(-1) # since prm[0] = prm[1]
        pickle.dump((prms, phiavgs, grads), open("change_prm_B{}_T{}.p".format(ds.B,ds.T), "wb"))
    plt.figure(figsize=[5,6])
    plt.plot(prms, phiavgs, 'k.', markersize=6)
    for prm, phiavg, grad in zip(prms, phiavgs, grads):
        plt.plot([prm-A, prm+A], [phiavg-grad*A, phiavg+grad*A], color='grey', linestyle='-')
    plt.ylabel('$\\rho(\Phi)$')
    plt.xlabel('$\gamma$')
    # plt.ylim(0.455,0.53)
    plt.tight_layout()
    plt.savefig("change_prm_B{}_T{}.png".format(ds.B,ds.T))
    plt.close()


def change_T():
    # convergence of gradient to different trajectory length, prm = prm[0] = prm[1]
    try:
        phiavgs, grads, nsegs = pickle.load( open("change_T.p", "rb"))
    except FileNotFoundError:
        nsegs = np.array([5, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2], dtype=int) 
        phiavgs = nanarray([nsegs.shape[0], n_repeat])
        sc, uc = nanarray([2, nsegs.shape[0], n_repeat, nprm])
        for i, nseg in enumerate(nsegs):
            print('\nnseg=',nseg)
            phiavgs[i], sc[i], uc[i] = wrapped_far(prm, nseg, W, n_repeat)
        grads = sc.sum(-1)
        grads = (sc-uc).sum(-1)
        pickle.dump((phiavgs, grads, nsegs), open("change_T.p", "wb"))
    
    plt.semilogx(nsegs, grads, 'k.')
    plt.xlabel('$A$')
    plt.ylabel('$\delta\\rho(\Phi)/\delta \gamma$')
    plt.tight_layout()
    plt.savefig('A_grad.png')
    plt.close()

    x = np.array([nsegs[0], nsegs[-1]])
    plt.loglog(nsegs, np.std(grads, axis=1), 'k.')
    plt.loglog(x, x**-0.5, 'k--')
    plt.xlabel('$A$')
    plt.ylabel('std $\delta\\rho(\Phi)/\delta \gamma$')
    plt.tight_layout()
    plt.savefig('A_std.png')
    plt.close()


def change_W():
    # gradient to different W, prm = prm[0] = prm[1]
    try:
        phiavgs, sc, uc, grads, Ws = pickle.load( open("change_W.p", "rb"))
    except FileNotFoundError:
        Ws = np.arange(10)
        phiavgs = nanarray([Ws.shape[0], n_repeat])
        sc, uc = nanarray([2, Ws.shape[0], n_repeat, nprm])
        for i, W in enumerate(Ws):
            print('\nW =',W)
            phiavgs[i], sc[i], uc[i] = wrapped_far(prm, nseg, W, n_repeat)
        grads = (sc-uc).sum(-1)
        pickle.dump((phiavgs, sc, uc, grads, Ws), open("change_W.p", "wb"))
    plt.plot(Ws, grads, 'k.')
    plt.plot([-1]*n_repeat, sc[0].sum(-1), 'k.') # W=-1: plot only sc, no uc
    plt.ylabel('$\delta\\rho(\Phi)/\delta \gamma$')
    plt.xlabel('$W$')
    plt.tight_layout()
    plt.savefig('W_grad.png')
    plt.close()


def change_W_std():
    # standard deviation to different W
    try:
        phiavgs, sc, uc, grads, Ws = pickle.load( open("change_W_std.p", "rb"))
    except FileNotFoundError:
        Ws = np.array([1e1, 2e1, 5e1, 1e2, 2e2, 5e2], dtype=int) 
        phiavgs = nanarray([Ws.shape[0], n_repeat])
        sc, uc = nanarray([2, Ws.shape[0], n_repeat, nprm])
        for i, W in enumerate(Ws):
            print('\nW =',W)
            phiavgs[i], sc[i], uc[i] = wrapped_far(prm, nseg, W, n_repeat)
        grads = (sc-uc).sum(-1)
        pickle.dump((phiavgs, sc, uc, grads, Ws), open("change_W_std.p", "wb"))

    x = np.array([Ws[0], Ws[-1]])
    plt.loglog(Ws, np.std(grads, axis=1), 'k.')
    plt.loglog(x, 0.005*x**0.5, 'k--')
    plt.xlabel('$W$')
    plt.ylabel('std $\delta\\rho(\Phi)/\delta \gamma$')
    plt.tight_layout()
    plt.savefig('W_std.png')
    plt.close()


def contours():
    # contourf, rho and sigma ~ J, gradient direction
    Nphi = 6
    Ngrad = 5

    try:
        prm, phiavg = pickle.load(open("contour_phi.p", "rb"))
    except FileNotFoundError:
        N = Nphi
        prm_ = np.linspace(-0.3, 0.3, N)
        prm = np.array(np.meshgrid(prm_, prm_)).transpose(1,2,0)
        phiavg = nanarray([N, N])
        for i in range(N):
            for j in range(N):
                print(i,j)
                phiii = wrapped_primal(prm[i,j], 1000, n_repeat)
                phiavg[i,j] = phiii.mean()
        pickle.dump((prm, phiavg), open("contour_phi.p", "wb"))

    fig = plt.figure(figsize=(10,8))
    ax = plt.axes()
    CS = plt.contourf(prm[...,0], prm[...,1], phiavg, 20, cmap=plt.cm.bone, origin='lower')
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('$\Phi_{avg}$')
    plt.xlabel('$\\gamma_1$')
    plt.ylabel('$\\gamma_2$')

    try:
        prm, grad = pickle.load( open("contour_grad.p", "rb"))
    except FileNotFoundError:
        N = Ngrad
        prm_ = np.linspace(-0.25, 0.25, N)
        prm = np.array(np.meshgrid(prm_, prm_)).transpose(1,2,0)
        grad = nanarray([N, N, nprm])
        for i in range(N):
            for j in range(N):
                print(i,j)
                _, sc, uc = wrapped_far(prm[i,j], nseg, W, n_repeat)
                grad[i,j] = (sc - uc).mean(axis=0)
        pickle.dump((prm, grad), open("contour_grad.p", "wb"))
    
    plt.scatter(prm[...,0], prm[...,1], color='k', s=15)
    Q = plt.quiver(prm[...,0], prm[...,1], grad[...,0], grad[...,1], units='x',
            pivot='tail', width=0.003, color='r', scale=10)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig('contour_rho_sig_J.png')
    plt.close()


if __name__ == '__main__': # pragma: no cover
    starttime = time.time()
    # graph()
    # trajectory()
    # all_info()
    # change_T()
    change_prm()
    # change_W()
    # change_W_std()
    # contours()
    print('prm=', ds.prm)
    endtime = time.time()
    print('time elapsed in seconds:', endtime-starttime)
