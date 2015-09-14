
import numpy as np
from GPy.core.parameterization.priors import Prior
import weakref


cachex = None
cachell = None
cachegrad = None

class GPyConstDiagonalGaussian(Prior):

    """
    Implementation of the multivariate Gaussian probability function, coupled with random variables.

    :param mu: mean (N-dimensional array)
    :param var: covariance matrix (NxN)

    .. Note:: Bishop 2006 notation is used throughout the code

    """
    from GPy.core.parameterization.domains import _REAL, _POSITIVE
    domain = _REAL
    _instances = []

    def __new__(cls, mu=0, var=1):  # Singleton:
        if cls._instances:
            cls._instances[:] = [instance for instance in cls._instances if instance()]
            for instance in cls._instances:
                if np.all(instance().mu == mu) and np.all(instance().var == var):
                    return instance()
        o = super(Prior, cls).__new__(cls, mu, var)
        cls._instances.append(weakref.ref(o))
        return cls._instances[-1]()

    def __init__(self, mu, var):
        self.mu = np.array(mu).flatten()
        self.var = float(var)
        self.input_dim = self.mu.size
        self.inv = 1.0/self.var

        self.constant = -0.5 * self.input_dim * np.log(2 * np.pi * self.var)



    def summary(self):
        raise NotImplementedError


    def pdf(self, x):
        return np.exp(self.lnpdf(x))


    def lnpdf(self, x):
        d = x - self.mu
        return self.constant - 0.5 * np.dot(d, d) * self.inv


    def lnpdf_grad(self, x):
        d = x - self.mu
        return -d*self.inv


    def rvs(self, n):
        return self.mu + np.random.randn(self.mu.size, n) * np.sqrt(self.var)


    def plot(self):
        import sys

        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import priors_plots

        priors_plots.multivariate_plot(self)

    def __getstate__(self):
        return self.mu, self.var

    def __setstate__(self, state):
        self.mu = state[0]
        self.var = float(state[1])
        self.input_dim = self.mu.size
        self.inv = 1.0/self.var
        self.constant = -0.5 * self.input_dim * np.log(2 * np.pi * self.var)


from GPy.kern import Matern32
from treegp.cover_tree import VectorTree
import pyublas
class MWrapperLLD(Matern32):
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Mat32LLD', distance_params=None):
        if distance_params is None:
            distance_params=np.array([30.0, 30.0,])
        tree_X = np.array([[0.0,] * input_dim,], dtype=float)
        self.ptree = VectorTree(tree_X, 1, "lld", distance_params,
                                "matern32", np.array([variance,], dtype=np.float))

        super(MWrapperLLD, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)


    def _scaled_dist(self, X, X2=None):
        XX1 = np.array(X, dtype=np.float64, order="C")
        if X2 is not None:
            XX2 = np.array(X2, dtype=np.float64, order="C")
        else:
            XX2 = XX1
        K = self.ptree.kernel_matrix(XX1, XX2, True)
        return K


    def gradients_X(self, dL_dK, X, X2=None):
        invdist = self._inv_dist(X, X2)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK

        sym = True
        XX1 = np.array(X, dtype=np.float64, order="C")
        if X2 is not None:
            XX2 = np.array(X2, dtype=np.float64, order="C")
            sym = False
        else:
            XX2 = XX1
        # now for each r, there is a deriv with respect to entries of X.
        dKv = np.zeros((XX2.shape[0],), dtype=np.float)
        ret = np.empty(X.shape, dtype=np.float64)

        for p in range(X.shape[0]):
            for i in range(X.shape[1]):
                self.ptree.dist_deriv_wrt_xi_row(XX1, XX2, p, i, dKv)
                if sym:
                    dKv[p]=0
                    ret[p, i] = 2*np.sum(dKv * dL_dr[p,:])
                else:
                    ret[p, i] = np.sum(dKv * dL_dr[p, :])

        import pdb; pdb.set_trace()
        return ret
