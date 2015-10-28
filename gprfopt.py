from gprf import GPRF
from block_clustering import Blocker
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

class SampledData(object):

    def __init__(self,
                 noise_var=0.01, n=30, ntrain=20, lscale=0.5,
                 obs_std=0.05, yd=10, seed=1):
        self.noise_var=noise_var
        self.n = n
        self.ntrain = ntrain
        self.lscale=lscale

        Xfull, Yfull, cov = sample_synthetic(n=n, noise_var=noise_var, yd=yd, lscale=lscale, seed=seed)
        self.cov = cov
        X, Y = Xfull[:ntrain,:], Yfull[:ntrain,:]
        self.Xtest, self.Ytest = Xfull[ntrain:,:], Yfull[ntrain:,:]

        self.SX, self.SY = X, Y
        self.block_idxs = None

        self.obs_std = obs_std
        np.random.seed(seed)
        self.X_obs = self.SX + np.random.randn(*X.shape)*obs_std

    def set_centers(self, centers):
        self.centers = np.asarray(centers)
        b = Blocker(self.centers)
        self.block_idxs = b.block_clusters(self.X_obs)
        self.reblock = lambda X : b.block_clusters(X)
        self.neighbors = b.neighbors(diag_connections=True)

    def cluster_rpc(self, blocksize):
        all_idxs = np.arange(self.ntrain)
        cluster_idxs, splits = cluster_rpc(self.X_obs, all_idxs, target_size=blocksize)
        self.block_idxs = cluster_idxs
        self.reblock = lambda X: cluster_rpc(X, all_idxs, target_size=blocksize, fixed_split=splits)[0]
        self.neighbors = None

    def build_gprf(self, X=None, cov=None, local_dist=1e-4):
        if X is None:
            X = self.X_obs # self.SX

        if cov is None:
            cov = self.cov
            noise_var = self.noise_var
        elif cov.shape[0]==1:
            noise_var = cov[0, 0]
            cov = GPCov(wfn_params=[cov[0,1]], dfn_params=cov[0,2:], dfn_str="euclidean", wfn_str="se")
        else:
            raise Exception("invalid cov params %s" % (cov))

        gprf = GPRF(X, Y=self.SY, block_fn = self.reblock,
                    block_idxs = self.block_idxs,
                    cov=cov, noise_var=noise_var, 
                    kernelized=False, 
                    neighbor_threshold=local_dist,
                    neighbors = self.neighbors if local_dist < 1.0 else [])
        return gprf

    def mean_distance(self, x):
        X = x.reshape(self.SX.shape)
        ds = np.linalg.norm(X-self.SX, axis=1)
        return np.mean(ds)

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
        return inferred_lscale/true_lscale

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


    def prediction_error(self, X=None, cov=None, local_dist=1.0):
        gprf = self.build_gprf(X=X, cov=cov, local_dist=local_dist)
        p = gprf.train_predictor()
        
        test_blocks = self.reblock(self.Xtest)
        
        def gaussian_ll(Y, M, C):
            ntest, yd = Y.shape
            P = np.linalg.inv(C)
            R = Y-M
            ll = -.5 * np.sum(P *  np.dot(R, R.T))
            ll -= .5 * yd * np.linalg.slogdet(C)[1]
            ll -= .5 * yd * ntest * np.log(2*np.pi)
            return ll

        ll_block = 0
        ll_block_diag = 0
        se_block = 0
        for idxs in test_blocks:
            Xt = self.Xtest[idxs]
            Yt = self.Ytest[idxs]

            ntest, yd = Yt.shape
            """
            for i in range(ntest):
                xt = Xt[i:i+1,:]
                yt = Yt[i:i+1,:]
                m, c = p_local(xt, test_noise_var=sdata.noise_var)   
                ll_marginal += gaussian_ll(yt, m, c)
                se_marginal += np.sum((yt-m)**2)
            """
            PM, PC = p(Xt, test_noise_var=self.noise_var)
            ll_block += gaussian_ll(Yt, PM, PC)
            ll_block_diag += gaussian_ll(Yt, PM, np.diag(np.diag(PC)))
            se_block += np.sum((Yt-PM)**2)

        ntest, yd = self.Ytest.shape

        Ymean = np.mean(self.SY, axis=0)
        se_baseline = np.sum((self.Ytest - Ymean)**2)
        smse = se_block / se_baseline

        Ystd = np.std(self.SY, axis=0)
        ll_baseline = np.sum([np.sum(scipy.stats.norm(loc=Ymean[i], scale=Ystd[i]).logpdf(self.Ytest[:, i])) for i in range(yd)])
        mll_baseline = ll_baseline / (ntest * yd)

        msll_block = ll_block / (ntest * yd) - mll_baseline
        msll_block_diag = ll_block_diag / (ntest * yd) - mll_baseline

        return smse, msll_block, msll_block_diag

    def x_prior(self, xx):
        flatobs = self.X_obs.flatten()
        t0 = time.time()

        n = len(xx)
        r = (xx-flatobs)/self.obs_std
        ll = -.5*np.sum( r**2)- .5 *n * np.log(2*np.pi*self.obs_std**2)

        lderiv = np.array([-(xx[i]-flatobs[i])/(self.obs_std**2) for i in range(len(xx))]).flatten()
        t1 = time.time()
        return ll, lderiv

    def x_prior_block(self, i, xx):
        idxs = self.block_idxs[i]

        flatobs = self.X_obs[idxs].flatten()

        n = len(xx)
        r = (xx-flatobs)/self.obs_std

        ll = -.5*np.sum( r**2)- .5 *n * np.log(2*np.pi*self.obs_std**2)
        lderiv = np.array([-(xx[i]-flatobs[i])/(self.obs_std**2) for i in range(len(xx))]).flatten()

        return ll, lderiv


    def random_init(self, jitter_std=None):
        if jitter_std is None:
            jitter_std = self.obs_std
        return self.X_obs + np.random.randn(*self.X_obs.shape)*jitter_std


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
        sdata = SampledData(n=n, ntrain=ntrain, lscale=lscale, obs_std=obs_std, seed=seed, yd=yd, noise_var=noise_var)

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

from gpy_shims import GPyConstDiagonalGaussian

def do_gpy_gplvm(d, gprf, X0, C0, sdata, method, maxsec=3600,
                 parallel=False, gplvm_type="bayesian", num_inducing=100):

    import GPy

    dim = sdata.SX.shape[1]
    # adjust kernel lengthscale to match GPy's defn of the RBF kernel incl a -.5 factor
    k = GPy.kern.RBF(dim, ARD=0, lengthscale=np.sqrt(.5)*sdata.lscale, variance=1.0)
    if C0 is None:
        k.lengthscale.fix()
    k.variance.fix()

    XObs = sdata.X_obs.copy()

    p = GPyConstDiagonalGaussian(XObs.flatten(), sdata.obs_std**2)
    if gplvm_type=="bayesian":
        print "bayesian GPLVM with %d inducing inputs" % num_inducing
        m = GPy.models.BayesianGPLVM(sdata.SY, dim, X=X0, X_variance = np.ones(XObs.shape)*sdata.obs_std**2, kernel=k, num_inducing=num_inducing)
        #m.X.mean.set_prior(p)
    elif gplvm_type=="sparse":
        print "sparse non-bayesian GPLVM with %d inducing inputs" % num_inducing
        m = GPy.models.SparseGPLVM(sdata.SY, dim, X=X0, kernel=k, num_inducing=num_inducing)

        from GPy.core import  Param
        m.X = Param('latent_mean', X0)
        m.link_parameter(m.X, index=0)

        #m.X.set_prior(p)
    elif gplvm_type=="basic":
        print "basic GPLVM on full dataset"
        m = GPy.models.GPLVM(sdata.SY, dim, X=XObs, kernel=k)
        #m.X.set_prior(p)

    m.likelihood.variance = sdata.noise_var
    m.likelihood.variance.fix()


    nmeans = X0.size
    sstep = [0,]
    f_log = open(os.path.join(d, "log.txt"), 'w')
    t0 = time.time()
    def llgrad_wrapper(xx):
        XX = xx[:nmeans].reshape(X0.shape)

        xd = X0.shape[1]
        n_ix = num_inducing * xd
        IX = xx[nmeans:nmeans+n_ix].reshape((-1, xd))

        np.save(os.path.join(d, "step_%05d_X.npy" % sstep[0]), XX)
        np.save(os.path.join(d, "step_%05d_IX.npy" % sstep[0]), IX)
        ll, grad = m._objective_grads(xx)


        prior_ll, prior_grad = sdata.x_prior(xx[:nmeans])
        ll -= prior_ll
        grad[:nmeans] -= prior_grad
        
        if C0 is not None:
            print "lscale", np.exp(xx[-1])

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, -ll)
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, -ll))
        f_log.flush()

        sstep[0] += 1

        if time.time()-t0 > maxsec:
            raise OutOfTimeError

        return ll, grad

    x0 = m.optimizer_array
    bounds = None

    try:
        r = scipy.optimize.minimize(llgrad_wrapper, x0, jac=True, method=method, bounds=bounds, options = {"ftol": 1e-6, "maxiter": 200})
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
        # near-uniform prior on (logscale) covariance params
        mean = -1
        std = 10
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

    cov_scale = 5. # hack to better condition optimization of cov params
    if gradC:
        c0 = np.log(C0.flatten()) * cov_scale
    else:
        c0 = np.array(())
    full0 = np.concatenate([x0, c0])

    sstep = [0,]
    f_log = open(os.path.join(d, "log.txt"), 'w')
    t0 = time.time()


    def lgpllgrad(x):

        if time.time()-t0 > maxsec:
            raise OutOfTimeError

        xx = x[:len(x0)]
        xc = x[len(x0):] / cov_scale

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
            gC /= cov_scale

        grad = np.concatenate([gX.flatten(), gC.flatten()])

        print "%d %.2f %.2f" % (sstep[0], time.time()-t0, ll)
        f_log.write("%d %.2f %.2f\n" % (sstep[0], time.time()-t0, ll))
        f_log.flush()

        sstep[0] += 1

        return -ll, -grad

    bounds = None
    try:
        print "optimizing with %s" % method
        r = scipy.optimize.minimize(lgpllgrad, full0, jac=True, method=method, bounds=bounds, options={"ftol": 1e-6, "maxiter": 200})
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

        l1 = sdata.mean_distance(X.flatten())
        c1 = sdata.lscale_error(FC) if FC is not None else 0.00
        l2 = sdata.x_prior(X.flatten())[0]
        if predict:
            smse_local, msll_local_block, msll_local_diag = sdata.prediction_error(X=X, cov=FC, local_dist=1.0)
            if local_dist < 1.0:
                smse, msll_block, msll_diag = sdata.prediction_error(X=X, cov=FC, local_dist=local_dist)
            else:
                smse, msll_block, msll_diag = smse_local, msll_local_block, msll_local_diag
        else:
            smse, smse_local, msll_local_block, msll_block, msll_local_diag, msll_diag = 0., 0., 0., 0., 0., 0.
        s = "%d %.2f %.2f %.8f %.8f %.8f %.4f %.4f %.4f %.4f %.4f %.4f" % (step, times[i], lls[i], c1, l1, l2, smse_local, smse, msll_local_block, msll_block, msll_local_diag, msll_diag)
        print s
        results.write(s + "\n")

    X = sdata.SX
    l1 = sdata.mean_distance(X.flatten()) # = 0.0
    c1 = 0.0
    l2 = sdata.x_prior(X.flatten())[0]
    if predict:
        smse_local, msll_local_block, msll_local_diag = sdata.prediction_error(X=X, cov=None, local_dist=1.0)
        if local_dist < 1.0:
            smse, msll_block, msll_diag = sdata.prediction_error(X=X, cov=None, local_dist = local_dist)
        else:
            smse, msll_block, msll_diag = smse_local, msll_local_block, msll_local_diag
    else:
        smse, smse_local, msll_local_block, msll_block, msll_local_diag, msll_diag = 0., 0., 0., 0., 0., 0.

    results.flush()

    gprf = sdata.build_gprf(X=X, local_dist=local_dist)    
    ll1 = -np.inf
    try:
        if gprf.n_blocks > 1:
            ll1 = gprf.llgrad()[0]
    except:
        pass

    s = "trueX inf %.2f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (ll1, c1, l1, l2, smse_local, smse, msll_local_block, msll_block, msll_local_diag, msll_diag)
    print s
    results.write(s + "\n")
    results.close()


def grid_centers(nblocks):
    pmax = np.ceil(np.sqrt(nblocks))*2+1
    pts = np.linspace(0, 1, pmax)[1::2]
    centers = [np.array((xx, yy)) for xx in pts for yy in pts]
    return centers

def do_run(d, lscale, n, ntrain, nblocks, yd, seed=0,
           fullgp=False, method=None,
           obs_std=None, local_dist=1.0, maxsec=3600,
           task='x', analyze_only=False, analyze_full=False,
           init_seed=-1, parallel=False,
           noise_var=0.01, rpc_blocksize=-1, 
           gplvm_type=None, num_inducing=-1,
           init_true = False,):

    if rpc_blocksize==-1:
        centers = grid_centers(nblocks)
        print "gprf with %d blocks" % (len(centers))
    else:
        centers = None
        print "gprf with rpc blocksize %d" % rpc_blocksize

    if obs_std is None:
        obs_std = lscale/10

    data = sample_data(n=n, ntrain=ntrain, lscale=lscale, obs_std=obs_std, yd=yd, seed=seed, centers=centers, noise_var=noise_var, rpc_blocksize=rpc_blocksize)
    #if not run_gpy:
    gprf = data.build_gprf(local_dist=local_dist)

    if task=='x':
        if init_true:
            X0 = data.SX
            gprf.update_X(X0)
        else:
            X0 = data.X_obs
        C0 = None
    elif task == 'cov':
        X0 = None
        gprf.update_X(data.SX)

        if init_seed >= 0:
            np.random.seed(init_seed)
            C0 = np.exp(np.random.randn(1, 4)-1)
        else:
            C0 = np.array((0.01, 1.0, 0.05,  0.05)).reshape(1,-1)
    elif task =='xcov':
        X0 = data.X_obs
        if init_seed >= 0:
            np.random.seed(init_seed)
            C0 = np.exp(np.random.randn(1, 1)-1)
            X0 = X0 + np.random.randn(*X0.shape)*0.005
        else:
            lscale = gprf.cov.dfn_params[0]
            C0 = np.array((lscale)).reshape(1,1)
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



def build_run_name(args):
    try:
        ntrain, ntest, nblocks, lscale, obs_std, local_dist, yd, method, task, init_seed, noise_var, rpc_blocksize, seed, gplvm_type, num_inducing, init_true = (args.n+args.ntest, args.ntest, args.nblocks, args.lscale, args.obs_std, args.local_dist, args.yd, args.method, args.task, args.init_seed, args.noise_var, args.rpc_blocksize, args.seed, args.gplvm_type, args.num_inducing, args.init_true)
    except:
        defaults = { 'yd': 50, 'seed': 0, 'local_dist': 0.05, "method": 'l-bfgs-b', 'task': 'x', 'init_seed': -1, 'noise_var': 0.01, 'rpc_blocksize': -1, 'gplvm_type': "gprf", 'num_inducing': -1, 'init_true': False}
        defaults.update(args)
        args = defaults
        ntrain, ntest, nblocks, lscale, obs_std, local_dist, yd, method, task, init_seed, noise_var, rpc_blocksize, seed, gplvm_type, num_inducing, init_true = (args['n'], args['ntest'], args['nblocks'], args['lscale'], args['obs_std'], args['local_dist'], args['yd'], args['method'], args['task'], args['init_seed'], args['noise_var'], args['rpc_blocksize'], args['seed'], args['gplvm_type'], args['num_inducing'], args["init_true"])
    run_name = "%d_%d_%s_%.6f_%.6f_%.4f_%d_%s_%s_%d_%s_s%s_%s%d" % (ntrain+ntest, ntrain, "%d" % nblocks if rpc_blocksize==-1 else "%06d" % rpc_blocksize, lscale, obs_std, local_dist, yd, method, task, -9999 if init_true else init_seed, "%.4f" % noise_var, "%d" % seed, gplvm_type, num_inducing)
    return run_name

def exp_dir(args):
    run_name = build_run_name(args)
    exp_dir = os.path.join(EXP_DIR, run_name)
    mkdir_p(exp_dir)
    return exp_dir

def main():

    mkdir_p(EXP_DIR)

    parser = argparse.ArgumentParser(description='gprf_opt')
    parser.add_argument('--n', dest='n', type=int, help="number of points to locate")
    parser.add_argument('--ntest', dest='ntest', type=int, default=500, help="sample additional test points to evaluate predictive accuracy (not in paper)")
    parser.add_argument('--nblocks', dest='nblocks', default=1, type=int, help="divides the sampled points into a grid of this many blocks. May do strange things if number is not a perfect square. Mutually exclusive with rpc_blocksize. ")
    parser.add_argument('--rpc_blocksize', dest='rpc_blocksize', default=-1, type=int, help="divides the sampled points into blocks using recursive projection clustering, aiming for this target blocksize. Mutually exclusive with nblocks. ")
    parser.add_argument('--lscale', dest='lscale', type=float, help="SE kernel lengthscale for the sampled functions")
    parser.add_argument('--obs_std', dest='obs_std', type=float, help="std of Gaussian noise corrupting the X locations")
    parser.add_argument('--local_dist', dest='local_dist', default=1.0, type=float, help="when using RPC clustering, specifies minimum kernel value necessary to connect blocks in a GPRF (1.0 corresponds to local GPs). When using grids of blocks, any setting other than 1.0 yields a GPRF with neighbor connections. ")
    parser.add_argument('--method', dest='method', default="l-bfgs-b", type=str, help="any optimization method supported by scipy.optimize.minimize (l-bfgs-b)")
    parser.add_argument('--seed', dest='seed', default=0, type=int, help="seed for generating synthetic data")
    parser.add_argument('--yd', dest='yd', default=50, type=int, help="number of output dimensions to sample (50)")
    parser.add_argument('--maxsec', dest='maxsec', default=3600, type=int, help="maximum number of seconds to run the optimization (3600)")
    parser.add_argument('--task', dest='task', default="x", type=str, help="'x', 'cov', or 'xcov' to infer locations, kernel hyperparams, or both. (x)")
    parser.add_argument('--analyze', dest='analyze', default=False, action="store_true", help="just analyze existing saved inference results, don't do any new inference")
    parser.add_argument('--analyze_full', dest='analyze_full', default=False, action="store_true", help="do a fuller (slower) analysis that also computes predictive accuracy")
    parser.add_argument('--parallel', dest='parallel', default=False, action="store_true", help="run multiple threads for local/gprf blocks. be careful when combining this with multithreaded BLAS libraries.")
    parser.add_argument('--init_seed', dest='init_seed', default=-1, type=int, help="if >=0, initialize optimization to locations generated from this random seed.")
    parser.add_argument('--init_true', dest='init_true', default=False, action="store_true", help="initialize optimization at true X locations instead of observed locations (False)")
    parser.add_argument('--noise_var', dest='noise_var', default=0.01, type=float, help="variance of iid noise in synthetic Y values")
    parser.add_argument('--gplvm_type', dest='gplvm_type', default="gprf", type=str, help = "use 'sparse' or 'bayesian' for GPy inducing point comparison (gprf)")
    parser.add_argument('--num_inducing', dest='num_inducing', default=0, type=int, help="number of inducing points to use with sparse approximations")

    args = parser.parse_args()

    d = exp_dir(args)
    do_run(d=d, lscale=args.lscale, obs_std=args.obs_std, local_dist=args.local_dist, n=args.n+args.ntest, ntrain=args.n, nblocks=args.nblocks, yd=args.yd, method=args.method, rpc_blocksize=args.rpc_blocksize, seed=args.seed, maxsec=args.maxsec, analyze_only=args.analyze, analyze_full = args.analyze_full, task=args.task, init_seed=args.init_seed, noise_var=args.noise_var, parallel=args.parallel, gplvm_type=args.gplvm_type, num_inducing=args.num_inducing, init_true=args.init_true)

if __name__ == "__main__":
    main()
