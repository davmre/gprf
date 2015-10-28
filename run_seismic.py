from gprf import GPRF
from gprfopt import OutOfTimeError, load_log

from pdtree_clustering import pdtree_cluster
from synthetic import sample_y

from treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian, sort_morton
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

def dist_deg(loc1, loc2):
    """
    Compute the great circle distance between two point on the earth's surface
    in degrees.
    loc1 and loc2 are pairs of longitude and latitude
    >>> int(dist_deg((10,0), (20, 0)))
    10
    >>> int(dist_deg((10,0), (10, 45)))
    45
    >>> int(dist_deg((-78, -12), (-10.25, 52)))
    86
    >>> dist_deg((132.86521, -0.45606493), (132.86521, -0.45606493)) < 1e-4
    True
    >>> dist_deg((127.20443, 2.8123965), (127.20443, 2.8123965)) < 1e-4
    True
    """
    lon1, lat1 = loc1
    lon2, lat2 = loc2

    rlon1 = np.radians(lon1)
    rlat1 = np.radians(lat1)
    rlon2 = np.radians(lon2)
    rlat2 = np.radians(lat2)

    dist_rad = 2*np.arcsin( \
        np.sqrt( \
            np.sin((rlat1-rlat2)/2.0)**2 + \
            np.cos(rlat1)*np.cos(rlat2)*   \
            np.sin((rlon1-rlon2)/2.0)** 2) \
                      )
    return np.degrees(dist_rad)

AVG_EARTH_RADIUS_KM = 6371.0
def dist_km(loc1, loc2):
    """
    Returns the distance in km between two locations specified in degrees
    loc = (longitude, latitude)
    """
    lon1, lat1 = loc1
    lon2, lat2 = loc2


    d = np.radians(dist_deg(loc1, loc2)) * AVG_EARTH_RADIUS_KM

    return d

COL_TIME, COL_TIMEERR, COL_LON, COL_LAT, COL_SMAJ, COL_SMIN, COL_STRIKE, COL_DEPTH, COL_DEPTHERR = np.arange(9)



def cov_prior(c):
    means = np.array((-2.3, 0.0, 3.6, 3.6))
    std = 1.5


    r = (c-means)/std
    ll = -.5*np.sum( r**2)- .5 *len(c) * np.log(2*np.pi*std**2)
    lderiv = (-(c-means)/(std**2)).reshape((-1,))

    # penalize pathologically large lengthscales.
    # if the optimizer tries a very large lengthscale, we *should*
    # add edges between all blocks with range of that new lengthscale,
    # but this is a pain and would slow things down a lot, so instead
    # we just discourage the optimizer from doing this.
    c = c.reshape((-1,))
    if c[2] > 5:
        penalty = np.exp(70*(c[2] - 5))
        ll -= penalty
        lderiv[2] -= 70 * np.exp(70*(c[2] - 5))

    return ll, lderiv


def do_optimization(d, gprf, X0, C0, cov_prior, x_prior, maxsec=3600, parallel=False, sparse=False):
    gradX = (X0 is not None)
    gradC = (C0 is not None)

    depth_scale = 100
    X0[:, 2] /= depth_scale

    if gradX:
        x0 = X0.flatten()
    else:
        x0 = np.array(())

    lscale_scale = 5.0
    if gradC:
        c0 = np.log(C0.flatten()) 
        #c0[2:] *= lscale_scale
    else:
        c0 = np.array(())
    full0 = np.concatenate([x0, c0])

    sstep = [0,]
    f_log = open(os.path.join(d, "log.txt"), 'w')
    t0 = time.time()

    cfname = os.path.join(d, "covs.txt")
    covf = open(cfname, 'w')



    def lgpllgrad(x):

        xx = x[:len(x0)]
        xc = x[len(x0):]

        XX = xx.reshape(X0.shape).copy()
        XX[:, 2] *= depth_scale
        if gradX:
            gprf.update_X(XX)
            np.save(os.path.join(d, "step_%05d_X.npy" % sstep[0]), XX)
        if gradC:
            #xc[2:]/= lscale_scale
            XC = xc.reshape(C0.shape)
            FC = np.exp(XC)
            FC[0,1] = 1.0 # don't learn sv
            if FC[0,0] > 10.0:
                FC[0,0] = 10.0
            if FC[0,2] > 999:
                FC[0,2] = 999
            elif FC[0,2] < 1.0:
                FC[0,2] = 1.0
            if FC[0,3] > 999:
                FC[0,3] = 999
            elif FC[0,3] < 1.0:
                FC[0,3] = 1.0
            
            gprf.update_covs(FC)
            np.save(os.path.join(d, "step_%05d_cov.npy" % sstep[0]), FC)

        try:
            ll, gX, gC = gprf.llgrad(local=True, grad_X=gradX, grad_cov=gradC,
                                     parallel=parallel, sparse=sparse)
        except Exception as e:
            print "fail", e
            return 1e10, np.random.randn(*x.shape)
        
        gX[:, 2] *= depth_scale

        gXl = np.linalg.norm(gX[:, 0])
        gXd = np.linalg.norm(gX[:, 2])
        print "gradient norm lon %f, depth %f, cov %s" % (gXl, gXd, gC.flatten())

        if gradX:
            prior_ll, prior_grad = x_prior(XX)
            prior_grad[:, 2] *= depth_scale
            ll += prior_ll
            gX = gX.flatten() + prior_grad.flatten()
        if gradC:
            prior_ll, prior_grad = cov_prior(xc)
            ll += prior_ll
            gC = (gC * FC).flatten() + prior_grad

            gC[1] = 0.0 # don't learn sv

            max_grad = np.max(np.abs(gC[2:]))
            if max_grad > 10:
                gC[2:] *= 2./(1  + max_grad/10.) 

        grad = np.concatenate([gX.flatten(), gC.flatten()])

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll),
        if gradC:
            print FC
        else:
            print
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, ll))
        f_log.flush()

        if gradC:
            covf.write("%d %s\n" % (sstep[0], FC))
            covf.flush()

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            raise OutOfTimeError


        return -ll, -grad

    try:
        bounds = None
        r = scipy.optimize.minimize(lgpllgrad, full0, jac=True, method="l-bfgs-b", bounds=bounds)
        rx = r.x
    except OutOfTimeError:
        print "terminated optimization for time"

    t1 = time.time()
    f_log.write("optimization finished after %.fs\n" % (time.time()-t0))
    f_log.close()

    covf.close()

    with open(os.path.join(d, "finished"), 'w') as f:
        f.write("")
        

def seismic_exp_dir(args):
    npts, block_size, thresh, init_cov, init_x, task, synth_lscale, obs_std = args.npts, args.rpc_blocksize, args.threshold, args.init_cov, args.init_x, args.task, args.synth_lscale, args.obs_std
    import hashlib
    base_dir = os.path.join(os.environ["HOME"], "seismic_experiments")
    init_str = "default"
    if init_cov or init_x:
        init_str = "_%s" % hashlib.md5(init_cov+init_x).hexdigest()[:8]
    run_name = "%d_%d_%.4f_%s_%s_%.0f_%.1f" % (npts, block_size, thresh, init_str, task, synth_lscale, obs_std)
    d =  os.path.join(base_dir, run_name)
    mkdir_p(d)
    return d

def dist_lld(x1, x2):
    d1 = dist_km((x1[0], x1[1]), (x2[0], x2[1]))
    d2 = x1[2] - x2[2]
    return np.sqrt(d1**2 + d2**2)

def analyze_run_result(args, gprf, x_prior, X_true, cov_true, lscale_true):
    d = seismic_exp_dir(args)
    seed = args.seed
    block_size = args.rpc_blocksize
    threshold = args.threshold
    init_cov = args.init_cov
    init_x = args.init_x
    npts = args.npts
    task = args.task

    steps, times, lls = load_log(d)

    rfname = os.path.join(d, "results.txt")
    results = open(rfname, 'w')
    print "writing results to", rfname


    def mad(X1, X2):
        n = X1.shape[0]
        dists = [dist_lld(X1[i], X2[i]) for i in range(n)]
        return np.mean(dists), np.median(dists)

    for i, step in enumerate(steps):

        try:
            fname_X = os.path.join(d, "step_%05d_X.npy" % step)
            X = np.load(fname_X)
        except IOError:
            X = sdata.SX

        try:
            fname_cov = os.path.join(d, "step_%05d_cov.npy" % step)
            FC = np.load(fname_cov)
        except IOError:
            FC = None

        if FC is not None:
            c1 = FC[0,2] / lscale_true
        else:
            c1 = 1.0
        l1, l2 = mad(X_true, X)

        s = "%d %.2f %.2f %.8f %.8f %.8f" % (step, times[i], lls[i], c1, l1, l2)
        print s
        results.write(s + "\n")

    gprf.update_X(X_true)
    gprf.update_covs(cov_true)
    lltrue = gprf.llgrad(grad_X=False, grad_cov=False)[0]
    priortrue = x_prior(X_true)[0]
    s = "true X ll %.2f" % (lltrue  + priortrue)
    print s
    results.write(s + "\n")
    results.close()

def load_data(synth_lscale, seed):
    # load file generated by seismic/generate_sorted.py
    sorted_isc = np.load("sorted_isc.npy")

    np.random.seed(seed)
    XX = sorted_isc[:, [COL_LON, COL_LAT, COL_DEPTH]].copy()
    y_fname = "seismic_Y_%.1f_%d.npy" % (synth_lscale, seed)
    try:
        SY = np.load(y_fname)
        cov = GPCov(wfn_params=[1.0], dfn_params=[synth_lscale,synth_lscale], dfn_str="lld", wfn_str="matern32")
    except:
        cov = GPCov(wfn_params=[1.0], dfn_params=[synth_lscale,synth_lscale], dfn_str="lld", wfn_str="matern32")
        SY = sample_y(XX, cov, 0.1, 50, sparse_lscales=6.0)
        np.save(y_fname, SY)
        print "sampled Y, saved to", y_fname

    return sorted_isc, SY, cov


def main():

    parser = argparse.ArgumentParser(description='seismic')

    parser.add_argument('--npts', dest='npts', default=-1, type=int, help="do inference on a subset of data, for debugging")
    parser.add_argument('--obs_std', dest='obs_std', default=-1, type=float, help="stddev for sampling observed X values")
    parser.add_argument('--threshold', dest='threshold', default=1.0, type=float, help="covariance threshold for adding a GPRF edge. 1.0 is local GPs, 0.6 is approx. one lengthscale.")
    parser.add_argument('--synth_lscale', dest='synth_lscale', default=40.0, type=float, help="Matern kernel lengthscale for generating Y values")
    parser.add_argument('--seed', dest='seed', default=0, type=int, help="seed for sampling ")
    parser.add_argument('--maxsec', dest='maxsec', default=3600, type=int, help="maximum number of seconds to run inference (3600)")
    parser.add_argument('--sparse', dest='sparse', default=False, action="store_true", help="use sparse rather than dense linear algebra for GP operations on each block. note this is NOT the same sense of sparsity as in inducing point methods. (False)")
    parser.add_argument('--analyze', dest='analyze', default=False, action="store_true", help="do no inference; just generate results from currently saved inference state")
    parser.add_argument('--rpc_blocksize', dest='rpc_blocksize', default=300, type=int, help="partition points into blocks of (no more than) this size.")
    parser.add_argument('--init_cov', dest='init_cov', default="", type=str, help="initialize with cov params from a file (.npy)")
    parser.add_argument('--init_x', dest='init_x', default="", type=str, help="initialize with X locations from a file (.npy)")
    parser.add_argument('--task', dest='task', default="xcov", type=str, help="'x', 'cov', or 'xcov' to infer locations, cov params, or both")
    parser.add_argument('--parallel', dest='parallel', default=False, action="store_true", help="use multiple threads to process blocks in parallel.")

    args = parser.parse_args()
    d = seismic_exp_dir(args)
    seed = args.seed
    block_size = args.rpc_blocksize
    threshold = args.threshold
    obs_std = args.obs_std
    init_cov = args.init_cov
    init_x = args.init_x
    npts = args.npts
    task = args.task
    analyze = args.analyze
    synth_lscale = args.synth_lscale

    sorted_isc, SY, cov = load_data(synth_lscale, seed)

    np.random.seed(seed)
    cov_true = np.array([0.1, cov.wfn_params[0], cov.dfn_params[0], cov.dfn_params[1]]).reshape((1, -1))
    if synth_lscale < 0:
        cov_true[0,0] = 1.0
        cov_true[0,1] = 0.1


    if args.npts > 0:
        npts = args.npts
        base = 60000
        sorted_isc = sorted_isc[base:base+npts, :]
        SY = SY[base:base+npts, :]
    else:
        npts = len(SY)
    
    X_true = sorted_isc[:, (COL_LON, COL_LAT, COL_DEPTH)]
    np.random.seed(seed)
    prior_std = obs_std * np.array([.01, .01, 1.])
    noise = np.random.randn(*X_true.shape) * prior_std
    means = X_true + noise
    X0 = means.copy()
    def x_prior(X):
        r = (X-means)/prior_std
        r2 = r/prior_std
        r = r.flatten()
        r2 = r2.flatten()
        n = X.shape[0]
        ll = -.5*np.sum( r**2)- .5 *n * (3*np.log(2*np.pi) +np.sum(np.log(prior_std**2)))
        lderiv = -r2.reshape(X.shape)
        return ll, lderiv            

    n = X0.shape[0]
    all_idxs = np.arange(n)
    cluster_idxs, reblock = pdtree_cluster(X0, blocksize=block_size)

    neighbor_fname = "neighbors_%d_%d_%.3f_%.3f.npy" % (n, block_size, threshold, obs_std)
    if threshold == 1.0:
        neighbors = []
    else:
        try:
            # it's possible to get neighbors directly from the principle axis tree,
            # but since it's a onetime cost we'll just bruteforce it rather than
            # bother with the implementation
            neighbors = np.load(neighbor_fname)
        except:
            neighbors = None

    if init_cov == "":
        C0 = cov_true.copy() 
    else:
        C0 = np.load(init_cov)

    if init_x == "":
        pass
    else:
        X0 = np.load(init_x)

    nv = cov_true[0,0]
    gprf =  GPRF(X0, SY, reblock, cov, nv,
                 neighbor_threshold=threshold,
                 block_idxs=cluster_idxs, neighbors=neighbors)
    if neighbors is None:
        np.save(neighbor_fname, gprf.neighbors)

    if task=="x":
        C0 = None
    elif task=="cov":
        X0 = None

    if not analyze:
        do_optimization(d, gprf, X0, C0, cov_prior, x_prior, maxsec=args.maxsec, parallel=args.parallel, sparse=args.sparse)

    if task=="x" or task=="xcov":
        analyze_run_result(args, gprf, x_prior, X_true, cov_true, synth_lscale)


if __name__ == "__main__":
    main()
