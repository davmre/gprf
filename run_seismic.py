from gprf import GPRF, Blocker
from gprfopt import cluster_rpc, OutOfTimeError, load_log
from seismic.seismic_util import scraped_to_evid_dict
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

#COL_IDX, COL_EVID, COL_LON, COL_LAT, COL_SMAJ, COL_SMIN, COL_STRIKE, COL_DEPTH, COL_DEPTHERR = np.arange(9)
COL_TIME, COL_TIMEERR, COL_LON, COL_LAT, COL_SMAJ, COL_SMIN, COL_STRIKE, COL_DEPTH, COL_DEPTHERR = np.arange(9)

def load_seismic_locations():
    fname = os.path.join(os.environ['HOME'],"aligned_data.npy")
    fnameY = os.path.join(os.environ['HOME'], "aligned_Y.npy")
    d=  np.load(fname)
    Y=  np.load(fnameY)
    return d, Y

def select_subset(full_data, Y, npts, center=None):
    if center is None:
        center = (135.73395, -3.6592229)

    n = full_data.shape[0]
    distances = [dist_km(center, (full_data[i, COL_LON], full_data[i, COL_LAT])) for i in range(n)]
    perm = np.array(sorted(np.arange(n), key = lambda i : distances[i]))
    selected = perm[:npts]
    return full_data[selected], Y[selected]

def load_kernel(fname, sd):
    fullK = np.memmap(fname, dtype='float32', mode='r', shape=(20473, 20473))

    n = sd.shape[0]
    myK = np.zeros((n,n), np.float32)
    for i, sd1 in enumerate(sd):
        idx1 = sd1[COL_IDX]
        myK[i,i] = 1.0
        for j, sd2 in enumerate(sd[:i]):
            idx2 = sd2[COL_IDX]
            min_idx = min(idx1, idx2)
            max_idx = max(idx1, idx2)
            k = fullK[max_idx, min_idx]
            if not np.isfinite(k):
                k = 0.0
            myK[i,j] = k
            myK[j, i] = k
    return myK

def cov_prior(c):
    means = np.array((-2.3, 0.0, 3.6, 3.6))
    std = 1.5

    r = (c-means)/std
    ll = -.5*np.sum( r**2)- .5 *len(c) * np.log(2*np.pi*std**2)
    lderiv = -(c-means)/(std**2)
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

    if gradC:
        c0 = np.log(C0.flatten())
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
            FC = np.exp(xc.reshape(C0.shape))
            gprf.update_covs(FC)
            np.save(os.path.join(d, "step_%05d_cov.npy" % sstep[0]), FC)

        ll, gX, gC = gprf.llgrad(local=True, grad_X=gradX, grad_cov=gradC,
                                       parallel=parallel, sparse=sparse)
        
        gX[:, 2] *= depth_scale

        gXl = np.linalg.norm(gX[:, 0])
        gXd = np.linalg.norm(gX[:, 2])
        print "gradient norm lon %f, depth %f" % (gXl, gXd)

        if gradX:
            prior_ll, prior_grad = x_prior(XX)
            prior_grad[:, 2] *= depth_scale
            ll += prior_ll
            gX = gX.flatten() + prior_grad.flatten()
        if gradC:
            prior_ll, prior_grad = cov_prior(xc)
            ll += prior_ll
            gC = (gC * FC).flatten() + prior_grad

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
        bounds = [(None, None),]*len(x0) + [(-12, 8)]*len(c0)
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


def get_idc_as_sd():
    homedir = os.getenv("HOME")
    idc_dict = scraped_to_evid_dict(os.path.join(homedir, "python/gprf/idc.txt"))
    isc_dict = scraped_to_evid_dict(os.path.join(homedir, "python/gprf/isc.txt"))

    sd = []
    for evid in isc.keys():
        sd.append(idc_dict[evid])
    return sd
        
def build_prior(sd):

    def azimuth_vector(azi_deg):
        angle_deg = 90 - azi_deg
        angle_rad = angle_deg * np.pi/180.0
        return np.array((np.cos(angle_rad), np.sin(angle_rad)))

    def normal_vec(v1):
        return np.array((-v1[1], v1[0]))

    def cov_matrix_km(azi_deg, var1, var2):
        v1 = azimuth_vector(azi_deg).reshape((-1, 1))
        v2 = normal_vec(v1).reshape((-1, 1))
        V = np.hstack((v1, v2))
        D = np.diag((var1, var2))
        return np.dot(V, np.dot(D, V.T))

    def conf_to_var(x, confidence=0.9):
        one_sided = 1.0 - (1.0-confidence)/2
        z = scipy.stats.norm.ppf(one_sided)
        stddev = x/z
        return stddev**2

    def linear_interpolate(x, keys, values):
        idx = np.searchsorted(keys, x, side="right")-1
        k1, k2 = keys[idx], keys[idx+1]
        v1, v2 = values[idx], values[idx+1]
        slope = (v2-v1)/(k2-k1)
        return v1 + (x-k1)*slope

    def degree_in_km(lat):
        # source: http://en.wikipedia.org/wiki/Latitude#Length_of_a_degree_of_latitude
        refs = np.array([0, 15, 30, 45, 60, 75, 90])
        lat_table = [110.574, 110.649, 110.852, 111.132, 111.412, 111.618, 111.694]
        lon_table = [111.320, 107.550, 96.486, 78.847, 55.800, 28.902, 0.000]
        alat = np.abs(lat)
        return linear_interpolate(alat, refs, lon_table), linear_interpolate(alat, refs, lat_table)

    def cov_km_to_deg(lat, C):
        klon, klat = degree_in_km(lat)
        L = np.diag((1.0/klon, 1.0/klat))
        return np.dot(L, np.dot(C, L.T))

    def uncertainty_to_cov(smaj, smin, strike, lat):
        #x1 = ellipse.max_horizontal_uncertainty
        #x2 = ellipse.min_horizontal_uncertainty
        #azi_deg = ellipse.azimuth_max_horizontal_uncertainty
        #C = cov_matrix_km(azi_deg, conf_to_var(x1), conf_to_var(x2))
        if smaj < 0.5:
            smaj = 0.5
        if smin < 0.5:
            smin = 0.5
        C = cov_matrix_km(strike, conf_to_var(smaj), conf_to_var(smin))
        CC = cov_km_to_deg(lat, C)
        return CC

    means = []
    covs = []
    for row in sd:
        means.append((row[COL_LON], row[COL_LAT], row[COL_DEPTH]))
        cov = np.zeros((3,3))
        cov[:2, :2] = uncertainty_to_cov(row[COL_SMAJ], row[COL_SMIN], row[COL_STRIKE], row[COL_LAT])

        deptherr = row[COL_DEPTHERR]
        if deptherr <= 20.0:
            deptherr = 40.0
        #if row[COL_DEPTH] == 0:
        #    print deptherr
        cov[2,2] = conf_to_var(deptherr)

        #cov *= 4
        covs.append(cov)
    means = np.array(means)

    precs = [np.linalg.inv(cov) for cov in covs]
    logdets = [np.linalg.slogdet(cov)[1] for cov in covs]

    def event_prior_llgrad(X):
        R = X - means
        rlons = R[:, 0]
        R[:, 0] = ((rlons + 180) % 360) - 180
        grad = np.zeros(R.shape)
        ll = 0
        for i, row in enumerate(R):
            alpha = np.dot(precs[i], row)
            rowll = 0
            rowll -= .5 * np.dot(row, alpha)
            rowll -= .5 * logdets[i]
            rowll -= .5 * np.log(2*np.pi)
            grad[i, :] = -alpha
            ll += rowll
            #if rowll < -100:
            #    import pdb; pdb.set_trace()
        return ll, grad

    shallow = means[:, 2] < 40
    dd = means.copy()
    n = np.sum(shallow)
    dd[shallow, 2] = np.random.rand(n) * 40
    #import pdb; pdb.set_trace()

    return dd, event_prior_llgrad

def seismic_exp_dir(args):
    npts, block_size, thresh, init_cov, init_x, task, synth_lscale, obs_std = args.npts, args.rpc_blocksize, args.threshold, args.init_cov, args.init_x, args.task, args.synth_lscale, args.obs_std
    import hashlib
    base_dir = "seismic_experiments"
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
    try:
        sorted_idc = np.load("sorted_idc.npy")
        sorted_isc = np.load("sorted_isc.npy")
        sorted_evids = np.load("sorted_evids.npy")
    except IOError:
        homedir = os.getenv("HOME")
        idc_dict = scraped_to_evid_dict(os.path.join(homedir, "python/gprf/idc.txt"))
        isc_dict = scraped_to_evid_dict(os.path.join(homedir, "python/gprf/isc.txt"))

        idc = []
        isc = []
        keyerror = 0
        evids = []
        for evid in isc_dict.keys():
            try:
                idc.append(idc_dict[evid])
                isc.append(isc_dict[evid])
                evids.append(evid)
            except KeyError:
                keyerror += 1
        idc = np.asarray(idc)
        isc = np.asarray(isc)
        evids = np.asarray(evids)
        n = len(idc)

        dists = np.asarray([dist_lld(idc[i, (COL_LON, COL_LAT, COL_DEPTH)], isc[i, (COL_LON, COL_LAT, COL_DEPTH)] ) for i in range(n)])
        inliers = dists < 3*idc[:, COL_SMAJ]
        idc = idc[inliers]
        isc = isc[inliers]

        XX = idc[:, [COL_LON, COL_LAT]]
        sorted_XX, sorted_idc, sorted_isc, sorted_evids = sort_morton(XX, idc, isc, evids)

        print "loaded X", len(idc), "from", n
        np.save("sorted_idc.npy", sorted_idc)
        np.save("sorted_isc.npy", sorted_isc)
        np.save("sorted_evids.npy", sorted_evids)

    #sorted_idc[:, COL_DEPTH] = sorted_isc[:, COL_DEPTH]

    np.random.seed(seed)
    XX = sorted_isc[:, [COL_LON, COL_LAT, COL_DEPTH]].copy()
    y_fname = "seismic_Y_%.1f_%d.npy" % (synth_lscale, seed)
    try:
        SY = np.load(y_fname)
        cov = GPCov(wfn_params=[1.0], dfn_params=[synth_lscale,synth_lscale], dfn_str="lld", wfn_str="matern32")
    except:
        if synth_lscale == -1:
            SY = load_fourier(sorted_evids)
        else:
            cov = GPCov(wfn_params=[1.0], dfn_params=[synth_lscale,synth_lscale], dfn_str="lld", wfn_str="matern32")
            SY = sample_y(XX, cov, 0.1, 50, sparse_lscales=6.0)

            print "sampled Y"
        np.save(y_fname, SY)

    if synth_lscale == -1:
        cov = GPCov(wfn_params=[1.0], dfn_params=[40.0,40.0], dfn_str="lld", wfn_str="matern32")

    return sorted_idc, sorted_isc, SY, cov

def load_fourier(evids):
    homedir = os.getenv("HOME")
    fourier = np.load(os.path.join(homedir, "python/gprf/fourier_signals.npy"))
    n = fourier.shape[0]
    evid_idx = {}
    for i in range(n):
        evid = int(fourier[i,0])
        evid_idx[evid] = i

    Y = []
    for evid in evids:
        idx = evid_idx[int(evid)]
        y = fourier[idx, 1:]
        if np.isnan(y).any():
            y[:] = 0
        Y.append(y)
    Y = np.asarray(Y)
    Y -= np.mean(Y, axis=0)
    Y /= np.std(Y)
    return Y

def main():

    parser = argparse.ArgumentParser(description='seismic')

    parser.add_argument('--npts', dest='npts', default=-1, type=int)
    parser.add_argument('--obs_std', dest='obs_std', default=-1, type=float)
    parser.add_argument('--threshold', dest='threshold', default=1.0, type=float)
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--maxsec', dest='maxsec', default=3600, type=int)
    parser.add_argument('--sparse', dest='sparse', default=False, action="store_true")
    parser.add_argument('--analyze', dest='analyze', default=False, action="store_true")
    parser.add_argument('--analyze_init', dest='analyze_init', default=False, action="store_true")
    parser.add_argument('--rpc_blocksize', dest='rpc_blocksize', default=300, type=int)
    parser.add_argument('--init_cov', dest='init_cov', default="", type=str)
    parser.add_argument('--init_x', dest='init_x', default="", type=str)
    parser.add_argument('--synth_lscale', dest='synth_lscale', default=40.0, type=float)
    #parser.add_argument('--prior', dest='prior', default="isc", type=str)
    parser.add_argument('--task', dest='task', default="x", type=str)
    parser.add_argument('--parallel', dest='parallel', default=False, action="store_true")

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

    sorted_idc, sorted_isc, SY, cov = load_data(synth_lscale, seed)

    np.random.seed(seed)
    cov_true = np.array([0.1, cov.wfn_params[0], cov.dfn_params[0], cov.dfn_params[1]]).reshape((1, -1))
    if synth_lscale < 0:
        cov_true[0,0] = 1.0
        cov_true[0,1] = 0.1


    if args.npts > 0:
        npts = args.npts
        base = 60000
        sorted_idc = sorted_idc[base:base+npts, :]
        sorted_isc = sorted_isc[base:base+npts, :]
        SY = SY[base:base+npts, :]
    else:
        npts = len(SY)
    
    X_true = sorted_isc[:, (COL_LON, COL_LAT, COL_DEPTH)]
    if obs_std < 0:
        X0, x_prior = build_prior(sorted_idc)
    else:
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

    if args.analyze_init:
        ll, _, _ = gprf.gaussian_llgrad(X0, gprf.Y)
        print "likelihood of initial state under true GP model:", ll
        import pdb; pdb.set_trace()
        sys.exit(0)

    if not analyze:
        do_optimization(d, gprf, X0, C0, cov_prior, x_prior, maxsec=args.maxsec, parallel=args.parallel, sparse=args.sparse)

    if task=="x" or task=="xcov":
        analyze_run_result(args, gprf, x_prior, X_true, cov_true, synth_lscale)


if __name__ == "__main__":
    main()
