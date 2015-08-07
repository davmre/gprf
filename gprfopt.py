from gprf import GPRF, Blocker
from synthetic import sample_synthetic

from treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian
from treegp.util import mkdir_p
import numpy as np
import scipy.stats
import scipy.optimize
import time
import os
import sys
import cPickle as pickle

import argparse

EXP_DIR = os.path.join(os.environ["HOME"], "gprf_experiments")


def sample_data(n, ntrain, lscale, obs_std, yd, seed,
                centers, noise_var, rpc_blocksize=-1):
    sample_basedir = os.path.join(os.environ["HOME"], "gprf_experiments", "synthetic_datasets")
    mkdir_p(sample_basedir)
    sample_fname = "%d_%d_%.6f_%.6f_%d_%d%s.pkl" % (n, ntrain, lscale, obs_std, yd, seed, "" if noise_var==0.01 else "_%.4f" % noise_var)
    sample_fname_full = os.path.join(sample_basedir, sample_fname)

    try:
        with open(sample_fname_full, 'rb') as f:
            sdata = pickle.load(f)
    except IOError:
        sdata = SampledData(n=n, ntrain=ntrain, lscale=lscale, obs_std=obs_std, seed=seed, centers=None, yd=yd, noise_var=noise_var)

        with open(sample_fname_full, 'wb') as f:
            pickle.dump(sdata, f)

    if centers is not None:
        sdata.set_centers(centers)
    else:
        np.random.seed(seed)
        sdata.cluster_rpc(rpc_blocksize)
    return sdata

class OutOfTimeError(Exception):
    pass


class SampledData(object):

    def __init__(self,
                 noise_var=0.01, n=30, ntrain=20, lscale=0.5,
                 obs_std=0.05, yd=10, centers=None, seed=1):
        self.noise_var=noise_var
        self.n = n
        self.ntrain = ntrain
        self.lscale=lscale

        Xfull, Yfull, cov = sample_synthetic(n=n, noise_var=noise_var, yd=yd, lscale=lscale, seed=seed)
        self.cov = cov
        X, Y = Xfull[:ntrain,:], Yfull[:ntrain,:]
        self.Xtest, self.Ytest = Xfull[ntrain:,:], Yfull[ntrain:,:]

        self.SX, self.SY = X, Y
        self.block_boundaries = [(0, X.shape[0])]
        self.centers = [np.array((0.0, 0.0))]

        self.obs_std = obs_std
        np.random.seed(seed)
        self.X_obs = self.SX + np.random.randn(*X.shape)*obs_std

    def set_centers(self, centers):
        b = Blocker(centers)
        self.SX, self.SY, self.perm, self.block_boundaries = b.sort_by_block(self.SX, self.SY)
        self.centers = centers
        self.X_obs = self.X_obs[self.perm]

    def cluster_rpc(self, blocksize):
        CC = cluster_rpc((self.SX, self.SY, np.arange(self.SX.shape[0])), target_size=blocksize)
        self.SX, self.SY, self.perm, self.block_boundaries = sort_by_cluster(CC)
        self.X_obs = self.X_obs[self.perm]

    def build_gprf(self, X=None, cov=None, local_dist=1e-4):
        if X is None:
            X = self.SX #self.X_obs

        if cov is None:
            cov = self.cov
            noise_var = self.noise_var
        elif cov.shape[0]==1:
            noise_var = cov[0, 0]
            cov = GPCov(wfn_params=[cov[0,1]], dfn_params=cov[0,2:], dfn_str="euclidean", wfn_str="se")
        else:
            raise Exception("invalid cov params %s" % (cov))

        gprf = GPRF(X, Y=self.SY, block_boundaries=self.block_boundaries,
                              cov=cov, noise_var=noise_var,
                              kernelized=False, neighbor_threshold=local_dist)
        return gprf

    def mean_abs_err(self, x):
        return np.mean(np.abs(x - self.SX.flatten()))

    def median_abs_err(self, x):
        X = x.reshape(self.SX.shape)
        R = X - self.SX
        d = np.sqrt(np.sum(R**2, axis=1))
        return np.median(d)

    def lscale_error(self, FC):
        true_lscale = self.cov.dfn_params[0]
        inferred_lscale = FC[0, 2]
        return np.abs(inferred_lscale-true_lscale)/true_lscale

    def prediction_error_gp(self, x):
        XX = x.reshape(self.X_obs.shape)
        ntest = self.n-self.ntrain
        ll = 0

        gp = GP(X=XX, y=self.SY[:, 0:1], cov_main=self.cov, noise_var=self.noise_var,
                sort_events=False, sparse_invert=False)
        pred_cov = gp.covariance(self.Xtest, include_obs=True)
        logdet = np.linalg.slogdet(pred_cov)[1]
        pred_prec = np.linalg.inv(pred_cov)

        for y, yt in zip(self.SY.T, self.Ytest.T):
            gp.y = y
            gp.alpha_r = gp.factor(y)
            pred_means = gp.predict(self.Xtest)
            rt = yt - pred_means

            lly =  -.5 * np.dot(rt, np.dot(pred_prec, rt))
            lly += -.5 * logdet
            lly += -.5 * ntest * np.log(2*np.pi)

            ll += lly

        return ll

    def prediction_error_bcm(self, X=None, cov=None, local_dist=1.0, marginal=False):
        ntest = self.n-self.ntrain
        yd = self.SY.shape[1]
        gprf = self.build_gprf(X=X, cov=cov, local_dist=local_dist)

        p = gprf.train_predictor()
        if marginal:
            ll = 0
            for xt, yt in zip(self.Xtest, self.Ytest):
                PM, PC = p(xt.reshape((1, -1)), test_noise_var=self.noise_var)
                PP = np.linalg.inv(PC)
                PR = np.reshape(yt, (1, -1))-PM
                ll -= .5 * np.sum(PP *  np.dot(PR, PR.T))
                ll -= .5 * yd * np.linalg.slogdet(PC)[1]
                ll -= .5 * yd * np.log(2*np.pi)
        else:
            PM, PC = p(self.Xtest, test_noise_var=self.noise_var)
            PP = np.linalg.inv(PC)
            PR = self.Ytest-PM

            ll =  -.5 * np.sum(PP *  np.dot(PR, PR.T))
            ll += -.5 * yd * np.linalg.slogdet(PC)[1]
            ll += -.5 * ntest * yd * np.log(2*np.pi)

        return ll / (ntest * yd)


    def x_prior(self, xx):
        flatobs = self.X_obs.flatten()
        t0 = time.time()

        n = len(xx)
        r = (xx-flatobs)/self.obs_std
        ll = -.5*np.sum( r**2)- .5 *n * np.log(2*np.pi*self.obs_std**2)

        lderiv = np.array([-(xx[i]-flatobs[i])/(self.obs_std**2) for i in range(len(xx))]).flatten()
        t1 = time.time()
        return ll, lderiv

    def random_init(self, jitter_std=None):
        if jitter_std is None:
            jitter_std = self.obs_std
        return self.X_obs + np.random.randn(*self.X_obs.shape)*jitter_std



from gpy_shims import GPyConstDiagonalGaussian

def do_gpy_gplvm(d, gprf, X0, C0, sdata, method, maxsec=3600,
                 parallel=False, gplvm_type="bayesian", num_inducing=100):

    import GPy

    dim = sdata.SX.shape[1]
    # adjust kernel lengthscale to match GPy's defn of the RBF kernel incl a -.5 factor
    k = GPy.kern.RBF(dim, ARD=0, lengthscale=np.sqrt(.5)*sdata.lscale, variance=1.0)
    k.lengthscale.fix()
    k.variance.fix()

    XObs = sdata.X_obs.copy()

    p = GPyConstDiagonalGaussian(XObs.flatten(), sdata.obs_std**2)
    if gplvm_type=="bayesian":
        print "bayesian GPLVM with %d inducing inputs" % num_inducing
        m = GPy.models.BayesianGPLVM(sdata.SY, dim, X=X0, X_variance = np.ones(XObs.shape)*sdata.obs_std**2, kernel=k, num_inducing=num_inducing)
        m.X.mean.set_prior(p)
    elif gplvm_type=="sparse":
        print "sparse non-bayesian GPLVM with %d inducing inputs" % num_inducing
        m = GPy.models.SparseGPLVM(sdata.SY, dim, X=X0, kernel=k, num_inducing=num_inducing)

        from GPy.core import  Param
        m.X = Param('latent_mean', X0)
        m.link_parameter(m.X, index=0)

        m.X.set_prior(p)
    elif gplvm_type=="basic":
        print "basic GPLVM on full dataset"
        m = GPy.models.GPLVM(sdata.SY, dim, X=XObs, kernel=k)
        m.X.set_prior(p)

    m.likelihood.variance = sdata.noise_var
    m.likelihood.variance.fix()


    nmeans = X0.size
    sstep = [0,]
    f_log = open(os.path.join(d, "log.txt"), 'w')
    t0 = time.time()
    def llgrad_wrapper(xx):
        XX = xx[:nmeans].reshape(X0.shape)

        np.save(os.path.join(d, "step_%05d_X.npy" % sstep[0]), XX)
        ll, grad = m._objective_grads(xx)

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, -ll)
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, -ll))
        f_log.flush()

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            raise OutOfTimeError

        return ll, grad

    def ll_wrapper(xx):
        XX = xx[:nmeans].reshape(X0.shape)

        np.save(os.path.join(d, "step_%05d_X.npy" % sstep[0]), XX)
        ll = m._objective(xx)

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll)
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, ll))
        f_log.flush()

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            raise OutOfTimeError

        return ll
        
    def grad_wrapper(xx):
        grad = m._grads(xx)
        return grad

    x0 = m.optimizer_array
    bounds = [(0.0, 1.0),]*nmeans + [(None, None)]*(x0.size-nmeans)


    try:
        if method=="scg":
            print "optimizating with SCG (WARNING: no bound constraints!)"
            from scg import SCG
            rx, flog, fe, status = SCG(ll_wrapper, grad_wrapper, x0)
            print status
        else:
            r = scipy.optimize.minimize(llgrad_wrapper, x0, jac=True, method=method, bounds=bounds)
            rx = r.x
    except OutOfTimeError:
        print "terminated optimization for time"


    t1 = time.time()
    f_log.write("optimization finished after %.fs\n" % (time.time()-t0))
    f_log.close()

    with open(os.path.join(d, "finished"), 'w') as f:
        f.write("")



def do_optimization(d, gprf, X0, C0, sdata, method, maxsec=3600, parallel=False):

    def cov_prior(c):
        mean = -1
        std = 3
        r = (c-mean)/std
        ll = -.5*np.sum( r**2)- .5 *len(c) * np.log(2*np.pi*std**2)
        lderiv = -(c-mean)/(std**2)
        return ll, lderiv

    def full_cov(C):
        if C.shape[1] == 1:
            # lscale
            FC = np.empty((C0.shape[0], 2+sdata.X_obs.shape[1]))
            FC[:, 0] = sdata.noise_var
            FC[:, 1] = 1.0
            FC[:, 2:3] = C
            FC[:, 3:4] = C
        elif C.shape[1] == 4:
            FC = C
        else:
            raise Exception("unrecognized cov param shape")
        return FC

    def collapse_cov_grad(grad_FC):
        if C0.shape[1] == 1:
            # lscale
            gradC = grad_FC[:, 2:3] + grad_FC[:, 3:4]
        elif C0.shape[1] == 4:
            gradC = grad_FC
        else:
            raise Exception("unrecognized cov param shape")
        return gradC

    gradX = (X0 is not None)
    gradC = (C0 is not None)

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


    def lgpll(x):
        global cachex, cachell, cachegrad
        if cachex is None or (x != cachex).any():
            cachex = x
            cachell, cachegrad = lgpllgrad(x)
        #else:
        #    print "cache hit ll"
        return cachell

    def lgpgrad(x):
        global cachex, cachell, cachegrad
        if (x != cachex).any():
            cachex = x
            cachell, cachegrad = lgpllgrad(x)
        #else:
        #    print "cache hit grad"
        return cachegrad

    def lgpllgrad(x):

        xx = x[:len(x0)]
        xc = x[len(x0):]

        if gradX:
            XX = xx.reshape(X0.shape)
            gprf.update_X(XX)
            np.save(os.path.join(d, "step_%05d_X.npy" % sstep[0]), XX)
        if gradC:
            C = np.exp(xc.reshape(C0.shape))
            FC = full_cov(C)
            print FC
            gprf.update_covs(FC)
            np.save(os.path.join(d, "step_%05d_cov.npy" % sstep[0]), FC)

        ll, gX, gC = gprf.llgrad(local=True, grad_X=gradX, grad_cov=gradC,
                                       parallel=parallel)

        if gradX:
            prior_ll, prior_grad = sdata.x_prior(xx)
            ll += prior_ll
            gX = gX.flatten() + prior_grad
        if gradC:
            prior_ll, prior_grad = cov_prior(xc)
            ll += prior_ll
            gC = (np.array(collapse_cov_grad(gC)) * C).flatten() + prior_grad

        grad = np.concatenate([gX.flatten(), gC.flatten()])

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll)
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, ll))
        f_log.flush()

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            raise OutOfTimeError

        return -ll, -grad

    bounds = [(0.0, 1.0),]*len(x0) + [(-10, 5)]*len(c0)
    try:

        if method=="scg":
            print "optimizating with SCG (WARNING: no bound constraints!)"
            from scg import SCG
            f = lgpll
            gradf = lambda x : lgpllgrad(x)[1]
            rx, flog, fe, status = SCG(lgpll, lgpgrad, full0)
            print status
        else:
            print "optimizing with %s" % method
            r = scipy.optimize.minimize(lgpllgrad, full0, jac=True, method=method, bounds=bounds)
            rx = r.x
    except OutOfTimeError:
        print "terminated optimization for time"

    t1 = time.time()
    f_log.write("optimization finished after %.fs\n" % (time.time()-t0))
    f_log.close()

    with open(os.path.join(d, "finished"), 'w') as f:
        f.write("")


def load_log(d):
    log = os.path.join(d, "log.txt")
    steps = []
    times = []
    lls = []
    with open(log, 'r') as lf:
        for line in lf:
            try:
                step, time, ll = line.split(' ')
                steps.append(int(step))
                times.append(float(time))
                lls.append(float(ll))
            except:
                continue

    return np.asarray(steps), np.asarray(times), np.asarray(lls)

def dump_covs(d):
    steps, times, lls = load_log(d)

    cov_fname = os.path.join(d, "covs.txt")
    print "writing to", cov_fname
    covs = open(cov_fname, 'w')

    for i, step in enumerate(steps):
        try:
            fname_cov = os.path.join(d, "step_%05d_cov.npy" % step)
            FC = np.load(fname_cov)

            s = "%d %s" % (step, FC)
            print s
            covs.write(s + "\n")
        except IOError:
            FC = None

    covs.close()

def analyze_run(d, sdata, local_dist=1.0, predict=False):

    steps, times, lls = load_log(d)

    rfname = os.path.join(d, "results.txt")

    results = open(rfname, 'w')
    print "writing results to", rfname

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

        l1 = sdata.mean_abs_err(X.flatten())
        c1 = sdata.lscale_error(FC) if FC is not None else 0.00
        l2 = sdata.x_prior(X.flatten())[0]
        if predict:
            p1 = sdata.prediction_error_bcm(X=X, cov=FC, local_dist=1.0)
            p2 = sdata.prediction_error_bcm(X=X, cov=FC, marginal=True)
        else:
            p1 = 0.0
            p2 = 0.0
        s = "%d %.2f %.2f %.8f %.8f %.8f %.4f %.4f" % (step, times[i], lls[i], c1, l1, l2, p1, p2)
        print s
        results.write(s + "\n")

    X = sdata.SX
    l1 = sdata.mean_abs_err(X.flatten()) # = 0.0
    c1 = 0.0
    l2 = sdata.x_prior(X.flatten())[0]
    if predict:
        p1 = sdata.prediction_error_bcm(X=X, cov=None, local_dist=1.0)
        p2 = sdata.prediction_error_bcm(X=X, cov=None, marginal=True)
    else:
        p1 = 0.0
        p2 = 0.0

    results.flush()

    gprf = sdata.build_gprf(X=X, local_dist=local_dist)    
    ll1 = -np.inf
    try:
        if gprf.n_blocks > 1:
            ll1 = gprf.llgrad()[0]
    except:
        pass

    s = "trueX inf %.2f %.4f %.4f %.4f %.4f %.4f" % (ll1, c1, l1, l2, p1, p2)
    print s
    results.write(s + "\n")
    results.close()

def sort_by_cluster(clusters):
    Xs, ys, perms = zip(*clusters)
    SX = np.vstack(Xs)
    SY = np.vstack(ys)
    perm = np.concatenate(perms).flatten()
    block_boundaries = []
    n = 0
    for (X,y,p) in clusters:
        cn = X.shape[0]
        block_boundaries.append((n, n+cn))
        n += cn
    return SX, SY, perm, block_boundaries

def cluster_rpc_kernelized(X, YY, target_size):
    n = X.shape[0]
    p = np.arange(n)
    CC = cluster_rpc((X, p.copy(), p.copy()), target_size)
    SX, SY, perm, block_boundaries = sort_by_cluster(CC)
    SYY = YY[perm, :][:, perm]
    return SX, SYY, perm, block_boundaries

def cluster_rpc((X, y, perm), target_size):
    n = X.shape[0]
    if n < target_size:
        return [(X, y, perm),]

    x1 = X[np.random.randint(n), :]
    x2 = x1
    while (x2==x1).all():
        x2 = X[np.random.randint(n), :]

    # what's the projection of x3 onto (x1-x2)?
    # imagine that x2 is the origin, so it's just x3 onto x1.
    # This is x1 * <x3, x1>/||x1||
    cx1 = x1 - x2
    nx1 = cx1 / np.linalg.norm(cx1)
    alphas = [ np.dot(xi-x2, nx1)  for xi in X]
    median = np.median(alphas)
    C1 = (X[alphas < median], y[alphas < median], perm[alphas < median])
    C2 = (X[alphas >= median], y[alphas >= median], perm[alphas >= median])

    L1 = cluster_rpc(C1, target_size=target_size)
    L2 = cluster_rpc(C2, target_size=target_size)
    return L1 + L2

def do_run(d, lscale, n, ntrain, nblocks, yd, seed=0,
           fullgp=False, method=None,
           obs_std=None, local_dist=1.0, maxsec=3600,
           task='x', analyze_only=False, analyze_full=False,
           init_seed=-1, parallel=False,
           noise_var=0.01, rpc_blocksize=-1, gplvm_type=None, num_inducing=-1):

    if rpc_blocksize==-1:
        pmax = np.ceil(np.sqrt(nblocks))*2+1
        pts = np.linspace(0, 1, pmax)[1::2]
        centers = [np.array((xx, yy)) for xx in pts for yy in pts]
        print "bcm with %d blocks" % (len(centers))
    else:
        centers = None
        print "bcm with rpc blocksize %d" % rpc_blocksize

    if obs_std is None:
        obs_std = lscale/10

    data = sample_data(n=n, ntrain=ntrain, lscale=lscale, obs_std=obs_std, yd=yd, seed=seed, centers=centers, noise_var=noise_var, rpc_blocksize=rpc_blocksize)
    #if not run_gpy:
    gprf = data.build_gprf(local_dist=local_dist)

    if task=='x':
        X0 = data.X_obs
        C0 = None
    elif task == 'cov':
        X0 = None
        if init_seed >= 0:
            np.random.seed(init_seed)
            C0 = np.exp(np.random.randn(1, 4)-1)
        else:
            C0 = np.array((0.1, 1.0, 0.3,  0.3)).reshape(1,-1)
    elif task =='xcov':
        X0 = data.X_obs
        if init_seed >= 0:
            np.random.seed(init_seed)
            C0 = np.exp(np.random.randn(1, 1)-1)
            X0 = X0 + np.random.randn(*X0.shape)*0.005
        else:
            C0 = np.array((0.3)).reshape(1,1)
    else:
        raise Exception("unrecognized task "+ task)

    if not analyze_only:
        if gplvm_type != "gprf":
            do_gpy_gplvm(d, gprf, X0, C0, data, method=method,
                         maxsec=maxsec, parallel=parallel,
                         gplvm_type=gplvm_type, num_inducing=num_inducing)
        else:
            do_optimization(d, gprf, X0, C0, data, method=method, maxsec=maxsec, parallel=parallel)


    analyze_run(d, data, local_dist=local_dist, predict=analyze_full)


def fast_analyze(d, sdata):

    segs = os.path.basename(d).split("_")
    ntrain = int(segs[0])
    n = int(segs[1])
    lscale = float(segs[3])
    obs_std = float(segs[4])
    yd = 50
    seed = int(segs[-1][1:])



    steps, times, lls = load_log(d)
    best_idx = np.argmax(lls)
    step = steps[best_idx]

    rfname = os.path.join(d, "fast_results.txt")
    results = open(rfname, 'w')

    fname_X = os.path.join(d, "step_%05d_X.npy" % step)
    X = np.load(fname_X)
    l1 = sdata.mean_abs_err(X.flatten())
    l2 = sdata.median_abs_err(X.flatten())

    s = "%.2f %.4f %.4f" % (times[best_idx], l1, l2)
    print s
    results.write(s + "\n")
    results.close()

def build_run_name(args):
    try:
        ntrain, n, nblocks, lscale, obs_std, local_dist, yd, method, task, init_seed, noise_var, rpc_blocksize, seed, gplvm_type, num_inducing = (args.ntrain, args.n, args.nblocks, args.lscale, args.obs_std, args.local_dist, args.yd, args.method, args.task, args.init_seed, args.noise_var, args.rpc_blocksize, args.seed, args.gplvm_type, args.num_inducing)
    except:
        defaults = { 'yd': 50, 'seed': 0, 'local_dist': 0.05, "method": 'l-bfgs-b', 'task': 'x', 'init_seed': -1, 'noise_var': 0.01, 'rpc_blocksize': -1, 'gplvm_type': "gprf", 'num_inducing': -1}
        defaults.update(args)
        args = defaults
        ntrain, n, nblocks, lscale, obs_std, local_dist, yd, method, task, init_seed, noise_var, rpc_blocksize, seed, gplvm_type, num_inducing = (args['ntrain'], args['n'], args['nblocks'], args['lscale'], args['obs_std'], args['local_dist'], args['yd'], args['method'], args['task'], args['init_seed'], args['noise_var'], args['rpc_blocksize'], args['seed'], args['gplvm_type'], args['num_inducing'])
    run_name = "%d_%d_%s_%.6f_%.6f_%.4f_%d_%s_%s_%d_%s_s%s_%s%d" % (ntrain, n, "%d" % nblocks if rpc_blocksize==-1 else "%06d" % rpc_blocksize, lscale, obs_std, local_dist, yd, method, task, init_seed, "%.4f" % noise_var, "%d" % seed, gplvm_type, num_inducing)
    return run_name

def exp_dir(args):
    run_name = build_run_name(args)
    exp_dir = os.path.join(EXP_DIR, run_name)
    mkdir_p(exp_dir)
    return exp_dir

def main():

    mkdir_p(EXP_DIR)

    MAXSEC=3600

    parser = argparse.ArgumentParser(description='gprf_opt')
    parser.add_argument('--ntrain', dest='ntrain', type=int)
    parser.add_argument('--n', dest='n', type=int)
    parser.add_argument('--nblocks', dest='nblocks', default=1, type=int)
    parser.add_argument('--rpc_blocksize', dest='rpc_blocksize', default=-1, type=int)
    parser.add_argument('--lscale', dest='lscale', type=float)
    parser.add_argument('--obs_std', dest='obs_std', type=float)
    parser.add_argument('--local_dist', dest='local_dist', default=1.0, type=float)
    parser.add_argument('--method', dest='method', default="l-bfgs-b", type=str)
    parser.add_argument('--seed', dest='seed', default=0, type=int)
    parser.add_argument('--yd', dest='yd', default=50, type=int)
    parser.add_argument('--maxsec', dest='maxsec', default=3600, type=int)
    parser.add_argument('--task', dest='task', default="x", type=str)
    parser.add_argument('--analyze', dest='analyze', default=False, action="store_true")
    parser.add_argument('--analyze_full', dest='analyze_full', default=False, action="store_true")
    parser.add_argument('--parallel', dest='parallel', default=False, action="store_true")
    parser.add_argument('--init_seed', dest='init_seed', default=-1, type=int)
    parser.add_argument('--noise_var', dest='noise_var', default=0.01, type=float)
    parser.add_argument('--gplvm_type', dest='gplvm_type', default="gprf", type=str)
    parser.add_argument('--num_inducing', dest='num_inducing', default=0, type=int)

    args = parser.parse_args()

    d = exp_dir(args)
    do_run(d=d, lscale=args.lscale, obs_std=args.obs_std, local_dist=args.local_dist, n=args.n, ntrain=args.ntrain, nblocks=args.nblocks, yd=args.yd, method=args.method, rpc_blocksize=args.rpc_blocksize, seed=args.seed, maxsec=args.maxsec, analyze_only=args.analyze, analyze_full = args.analyze_full, task=args.task, init_seed=args.init_seed, noise_var=args.noise_var, parallel=args.parallel, gplvm_type=args.gplvm_type, num_inducing=args.num_inducing)

if __name__ == "__main__":
    main()
