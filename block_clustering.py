import numpy as np


def pair_distances(Xi, Xj):
    return np.sqrt(np.outer(np.sum(Xi**2, axis=1), np.ones((Xj.shape[0]),)) - 2*np.dot(Xi, Xj.T) + np.outer((np.ones(Xi.shape[0]),), np.sum(Xj**2, axis=1)))

class Blocker(object):

    def __init__(self, block_centers):
        self.block_centers = block_centers
        self.n_blocks = len(block_centers)

    def get_block(self, X_new):
        dists = [np.linalg.norm(X_new - center) for center in self.block_centers]
        return np.argmin(dists)

    def block_clusters(self, X):
        dists = pair_distances(X, self.block_centers)
        blocks = np.argmin(dists, axis=1)
        idxs = []
        all_idxs = np.arange(len(X))
        for i in range(self.n_blocks):
            block_idxs = all_idxs[blocks == i]
            idxs.append(block_idxs)

        return idxs

    def neighbors(self, diag_connections=True):
        neighbors = []

        if len(self.block_centers) <= 1:
            return []

        center_distances = pair_distances(self.block_centers, self.block_centers)
        cc = center_distances.flatten()
        cc = cc[cc > 0]
        min_dist = np.min(cc) + 1e-6
        diag_dist = np.min(cc[cc > min_dist]) + 1e-6
        connect_dist = diag_dist if diag_connections else min_dist

        for i in range(self.n_blocks):
            for j in range(i):
                if center_distances[i,j] < connect_dist:
                    neighbors.append((i,j))
        return neighbors


def cluster_rpc(X, idxs, target_size, fixed_split=None):
    # given: 
    #  global array X listing all points
    #  list of indices idxs
    #  target cluster size
    # return:
    #  list of indices corresponding to a partition of the indices into the target size
    

    n = len(idxs)
    if fixed_split is not None and len(fixed_split) == 0:
        return [idxs,], ()

    if fixed_split is None:
        if n < target_size:
            return [idxs,], ()

        idx1 = np.random.choice(idxs)
        idx2 = idx1
        while (idx2==idx1).all():
            idx2 = np.random.choice(idxs)


        x1 = X[idx1,:]
        x2 = X[idx2,:]

        # what's the projection of x3 onto (x1-x2)?
        # imagine that x2 is the origin, so it's just x3 onto x1.
        # This is x1 * <x3, x1>/||x1||
        cx1 = x1 - x2
        nx1 = cx1 / np.linalg.norm(cx1)
        fs1 = None
        fs2 = None
    else:
        (nx1, x2), fs1, fs2 = fixed_split

    if n > 0:
        alphas = [ np.dot(X[i,:]-x2, nx1)  for i in idxs]
        median = np.median(alphas)
        idxs1 = idxs[alphas < median]
        idxs2 = idxs[alphas >= median]
    else:
        idxs1 = ()
        idxs2 = ()

    L1, split1 = cluster_rpc(X, idxs1, target_size=target_size, fixed_split=fs1)
    L2, split2 = cluster_rpc(X, idxs2, target_size=target_size, fixed_split=fs2)
    
    split = ((nx1, x2), split1, split2)

    return L1 + L2, split
