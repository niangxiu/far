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

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


# default parameters 
n_repeat = 4
ncpu = 2


def wrapped_far(prm, nseg, W, n_repeat): 
    ds.prm = prm
    ds.nseg = nseg
    ds.W = W
    if n_repeat == 1 :
        results = [far()]
    else:
        with Pool(processes=ncpu) as pool:
            results = pool.starmap(far)
    phiavg_, sc_, uc_, *_ = zip(*results)
    print('prm, nseg, W, phiavg, sc, uc, grad')
    [print('{:.2e}, {:d}, {:d}, {:.2e}, {:.2e}, {:.2e}, {:.3e}'.\
            format(ds.prm, ds.nseg, ds.W, phiavg, sc, uc, sc-uc)) \
            for phiavg, sc, uc in zip(phiavg_, sc_, uc_)]
    return np.array(phiavg_), np.array(sc_), np.array(uc_)


def change_T():
    # convergence of gradient to different trajectory length
    try:
        phiavgs, grads, nsegs = pickle.load( open("change_T.p", "rb"))
    except FileNotFoundError:
        nsegs = np.array([5, 1e1, 2e1, 5e1, 1e2, 2e2], dtype=int) 
        phiavgs, sc, uc = np.empty([3, nsegs.shape[0], n_repeat])
        for i, nseg in enumerate(nsegs):
            print('\nK=',nseg)
            phiavgs[i], sc[i], uc[i] = wrapped_far(prm, nseg, W, n_repeat)
        grads = sc-uc
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
    # gradient to different W
    try:
        phiavgs, sc, uc, grads, Ws = pickle.load( open("change_W.p", "rb"))
    except FileNotFoundError:
        Ws = np.arange(10)
        phiavgs, sc, uc = np.empty([3, Ws.shape[0], n_repeat])
        for i, W in enumerate(Ws):
            print('\nW =',W)
            phiavgs[i], sc[i], uc[i] = wrapped_far(prm, nseg, W, n_repeat)
        grads = sc-uc
        pickle.dump((phiavgs, sc, uc, grads, Ws), open("change_W.p", "wb"))
    plt.plot(Ws, grads, 'k.')
    plt.plot([-1]*n_repeat, sc[0], 'k.')
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
        phiavgs, sc, uc = np.empty([3, Ws.shape[0], n_repeat])
        for i, W in enumerate(Ws):
            print('\nW =',W)
            phiavgs[i], sc[i], uc[i] = wrapped_far(prm, nseg, W, n_repeat)
        grads = sc-uc
        pickle.dump((phiavgs, sc, uc, grads, Ws), open("change_W_std.p", "wb"))

    x = np.array([Ws[0], Ws[-1]])
    plt.loglog(Ws, np.std(grads, axis=1), 'k.')
    plt.loglog(x, 0.005*x**0.5, 'k--')
    plt.xlabel('$W$')
    plt.ylabel('std $\delta\\rho(\Phi)/\delta \gamma$')
    plt.tight_layout()
    plt.savefig('W_std.png')
    plt.close()


def change_prm():
    # grad for different prm
    n_repeat = 1 # must use 1, since prm in ds.py is fixed at the time the pool generates
    prms = np.linspace(-0.3, 0.4, 15)
    A = 0.015 # step size in the plot
    phiavgs, sc, uc = np.empty([3,prms.shape[0]])
    try:
        prms, phiavgs, grads = pickle.load(open("change_prm.p", "rb"))
    except FileNotFoundError:
        for i, prm in enumerate(prms):
            ds.prm = prm
            phiavgs[i], sc[i], uc[i] = wrapped_far(prm, nseg, W, n_repeat)
        grads = sc - uc
        pickle.dump((prms, phiavgs, grads), open("change_prm.p", "wb"))
    plt.plot(prms, phiavgs, 'k.', markersize=6)
    for prm, phiavg, grad in zip(prms, phiavgs, grads):
        plt.plot([prm-A, prm+A], [phiavg-grad*A, phiavg+grad*A], color='grey', linestyle='-')
    plt.ylabel('$\\rho(\Phi)$')
    plt.xlabel('$\gamma$')
    plt.tight_layout()
    plt.savefig('prm_obj.png')
    plt.close()


def all_info():
    # generate all info
    starttime = time.time()
    phiavg, sc, uc, u, v, Juv, vt = far(5, W)
    endtime = time.time()
    print('time for far:', endtime-starttime)
    for i, j in [[1,0], [1,2]]:
        plt.figure(figsize=[6,6])
        plt.plot(u[:,:,i].reshape(-1), u[:,:,j].reshape(-1), '.', markersize=1)
        plt.xlabel('$x^{}$'.format(i+1))
        plt.ylabel('$x^{}$'.format(j+1))
        plt.tight_layout()
        plt.savefig('x{}_x{}.png'.format(i+1, j+1))
        plt.close()
    print('phiavg, sc, uc, grad = ', '{:.3e}, {:.3e}, {:.3e}, {:.3e}'.format(phiavg, sc, uc, sc-uc))
    # print('Lyapunov exponenets = ', LEs)
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13,12))
    vn = v[:,:,0].reshape(-1)
    ax1.plot(np.arange(vn.shape[0]),vn)
    ax1.set_title('v norm')
    Juv = Juv[:,:].reshape(-1)
    ax2.plot(np.arange(vn.shape[0]),Juv)
    ax2.set_title('Ju @ v')
    ax3.scatter(u[:,:,0], v[:,:,0])
    vtn = vt[:,:,0].reshape(-1)
    ax4.plot(np.arange(vtn.shape[0]),vtn)
    ax4.set_title('vtilde norm')
    plt.tight_layout()
    plt.savefig('v norm.png')
    plt.close()


def trajectory():
    np.random.seed()
    starttime = time.time()
    u, J, Ju = primal_raw(preprocess(), 20*1000)
    endtime = time.time()
    print('time for compute trajectory:', endtime-starttime)
    u = u[1000:]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(u[:,0], u[:,1], u[:,2], '.', markersize=1)
    ax.view_init(70, 135)
    plt.savefig('3dview.png')
    plt.close()


if __name__ == '__main__': # pragma: no cover
    starttime = time.time()
    change_prm()
    # change_W()
    # change_W_std()
    # change_T()
    # trajectory()
    # all_info()
    print('prm=', ds.prm)
    endtime = time.time()
    print('time elapsed in seconds:', endtime-starttime)
