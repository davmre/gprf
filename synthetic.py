import numpy as np
from treegp.gp import GPCov, mcov

def sample_crazy_shape(seed, n, std=0.005):

    np.random.seed(seed)

    if seed % 1000 > 4:
        std = 0.27386127875258309 / np.sqrt(n)

    def sample_X(n=1000):
        X1 = sample_points_line(n/2, (0.1, 0.1), (0.9, 0.9))
        X2 = sample_points_line(n/2, (0.1, 0.9), (0.9, 0.1))
        return np.vstack([X1, X2])

    def sample_diamond(n=1000):
        X1 = sample_points_line(n/4, (0.5, 0.9), (0.9, 0.5))
        X2 = sample_points_line(n/4, (0.5, 0.9), (0.1, 0.5))
        X3 = sample_points_line(n/4, (0.1, 0.5), (0.5, 0.1))
        X4 = sample_points_line(n/4, (0.5, 0.1), (0.9, 0.5))
        return np.vstack([X1, X2, X3, X4])

    def sample_star(points=10, n=1000):
        Xs = []
        angles = (2*np.pi)/points
        for i in range(points):
            x1 = np.array((0.5, 0.5))
            theta = i * angles
            v = np.array((np.cos(theta), np.sin(theta)))
            v = 0.4 * v / np.linalg.norm(v)
            X1 = sample_points_line(n/4, x1, x1+v)
            Xs.append(X1)
        return np.vstack(Xs)

    def sample_crazy_lines(n=1000, std=0.005):
        seg_npts = 250
        segments = n / seg_npts
        segment_len = 41.10960958218894 / np.sqrt(n) # length 1.3 at 1000 pts

        Xs = []
        for i in range(segments):
            while True:
                x1 = np.random.rand(2)
                v = np.random.rand(2)
                v /= np.linalg.norm(v)
                x2 = x1 + v * segment_len
                if x2[0] > 0 and x2[0] < 1 and x2[1] > 0 and x2[1] < 1:
                    Xs.append(sample_points_line(seg_npts, x1, x2, std=std))
                    break
        return np.vstack(Xs)

    def sample_points_line(n, x1, x2, std=0.005):
        x2 = np.array(x2)
        x1 = np.array(x1)
        v = x2-x1
        rs = np.random.rand(n)
        pts = np.array([x1 + r*v for r in rs])
        X = pts + np.random.randn(*pts.shape) * std
        return X

    def sample_fault(n=1200, std=0.005):
        sn = n/10
        Xs = []
        x1 = np.array((0.1, 0.1))
        x2 = np.array((0.2, 0.2))
        Xs.append(sample_points_line(sn, x1, x2))
        x3 = np.array((0.2, 0.5))
        Xs.append(sample_points_line(sn, x2, x3))

        x4 = np.array((0.3, 0.3))
        Xs.append(sample_points_line(sn, x2, x4))

        x5 = np.array((0.5, 0.1))
        Xs.append(sample_points_line(sn, x4, x5))

        x6 = np.array((0.4, 0.45))
        Xs.append(sample_points_line(sn, x4, x6))
        x7 = np.array((0.2, 0.8))
        Xs.append(sample_points_line(sn, x6, x7))

        x8 = np.array((0.5, 0.6))
        Xs.append(sample_points_line(sn, x6, x8))
        x9 = np.array((0.9, 0.4))
        Xs.append(sample_points_line(sn, x8, x9))
        x10 = np.array((0.8, 0.9))
        Xs.append(sample_points_line(sn, x8, x10))
        x11 = np.array((0.8, 0.1))
        Xs.append(sample_points_line(sn, x9, x11))
        return np.vstack(Xs)

    if seed < 1100:
        return sample_fault(n=n)
    elif seed < 1200:
        return sample_X(n=n)
    elif seed < 1300:
        return sample_diamond(n=n)
    elif seed < 1350:
        return sample_crazy_lines(n=n, std=0.005)
    elif seed < 1400:
        return sample_crazy_lines(n=n, std=0.00005)


def sample_y(X, cov, noise_var, yd, sparse_lscales=4.0):
    n = X.shape[0]

    if n < 40000:
        from gpy_linalg import jitchol
        KK = mcov(X, cov, noise_var)
        n = KK.shape[0]

        L = jitchol(KK)
        #L = np.linalg.cholesky(KK)
        Z = np.random.randn(X.shape[0], yd)
        y = np.dot(L, Z)
    else:
        import scipy.sparse
        import scikits.sparse

        from treegp.cover_tree import VectorTree
        import pyublas

        n = X.shape[0]
        ptree = VectorTree(X, 1, cov.dfn_str, cov.dfn_params, cov.wfn_str, cov.wfn_params)

        entries = ptree.sparse_training_kernel_matrix(X, sparse_lscales, False)
        KKsparse = scipy.sparse.coo_matrix((entries[:,2], (entries[:,0], entries[:,1])), shape=(n,n), dtype=float)
        KKsparse = KKsparse + noise_var * scipy.sparse.eye(n)

        # attempt sparsity cause nothing else is going to work
        factor = scikits.sparse.cholmod.cholesky(KKsparse)
        L = factor.L()
        P = factor.P()
        Pinv = np.argsort(P)
        z = np.random.randn(n, yd)
        y = np.array((L * z)[Pinv])
    return y

def sample_synthetic(seed=1, n=400, xd=2, yd=10, lscale=0.1, noise_var=0.01):
    # sample data from the prior
    
    if seed < 1000:
        np.random.seed(seed)
        X = np.random.rand(n, xd)
    else:
        X = sample_crazy_shape(seed, n)
        assert(X.shape[0]==n)

    cov = GPCov(wfn_params=[1.0], dfn_params=[lscale, lscale], dfn_str="euclidean", wfn_str="se")

    y = sample_y(X, cov, noise_var, yd)

    return X, y, cov
