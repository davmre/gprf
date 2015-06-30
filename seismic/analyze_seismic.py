from gprf import GPRF
from run_seismic import load_seismic_locations, select_subset, dist_km

from treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian
from treegp.util import mkdir_p
import numpy as np
import scipy.stats
import scipy.optimize
import scipy.io
import time
import os
import sys
import cPickle as pickle
import argparse


COL_IDX, COL_EVID, COL_LON, COL_LAT, COL_SMAJ, COL_SMIN, COL_STRIKE, COL_DEPTH, COL_DEPTHERR = np.arange(9)



def load_XY(d):
    dname = os.path.basename(d)
    npts = int(dname.split("_")[0])
    block_size = int(dname.split("_")[1])
    ddd, Y = load_seismic_locations()
    dd, YY = select_subset(ddd, Y, npts)

    last_Xname = [fname for fname in sorted(os.listdir(d))[::-1] if fname.startswith("step") and fname.endswith("_X.npy")][0]
    perm = np.load(os.path.join(d, "perm.npy"))
    rperm = np.argsort(perm)
    SX = np.load(os.path.join(d, last_Xname))
    X = SX[rperm]
    return X, YY

def mean_distance(X1, X2):
    distances = [dist_km(x1[:2], x2[:2]) for x1, x2 in zip(X1, X2)]
    return np.mean(distances), np.median(distances)

def main():

    d1 = sys.argv[1]
    d2 = sys.argv[2]

    C0 = np.load("seismic_experiments/19980_200_1.0000_default_cov/step_00030_cov.npy")

    X1, Y1 = load_XY(d1)
    X2, Y2 = load_XY(d2)
    npts = len(X1)

    md, mdd = mean_distance(X1, X2)
    print "mean distance", md
    print "median distance", mdd

    """
    nv = C0[0,0]
    cov = GPCov(wfn_params=[C0[0,1]], dfn_params=C0[0, 2:], dfn_str="lld", wfn_str="matern32")
    gprf =  GPRF(X1, Y1, [(0, npts)],
                 cov, nv,
                 neighbor_threshold=0.0,
                 nonstationary=False,
                 kernelized=False)

    gpll, _, _ = gprf.gaussian_llgrad(X1, Y1)

    print "ll under full GP", gpll
    """
if __name__ == "__main__":
    main()
