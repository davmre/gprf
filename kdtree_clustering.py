import numpy as np
import sys

class KDTree(object):
    """
    slightly modified (and pared down) version of scipy's kdtree implementation. 
    The major change is to split at the median data point (instead of the spatial midpoint
    of the bounds) to guarantee that all leaves contain roughly the same number of points. 
    """
    def __init__(self, data, leafsize=10):
        self.data = np.asarray(data)
        self.n, self.m = np.shape(self.data)
        self.leafsize = int(leafsize)
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = np.amax(self.data,axis=0)
        self.mins = np.amin(self.data,axis=0)

        self.tree = self.__build(np.arange(self.n), self.maxes, self.mins)

    class node(object):
        if sys.version_info[0] >= 3:
            def __lt__(self, other):
                return id(self) < id(other)

            def __gt__(self, other):
                return id(self) > id(other)

            def __le__(self, other):
                return id(self) <= id(other)

            def __ge__(self, other):
                return id(self) >= id(other)

            def __eq__(self, other):
                return id(self) == id(other)

    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(idx)

    class innernode(node):
        def __init__(self, split_dim, split, less, greater):
            self.split_dim = split_dim
            self.split = split
            self.less = less
            self.greater = greater
            self.children = less.children+greater.children

    def __build(self, idx, maxes, mins):
        if len(idx) <= self.leafsize:
            if np.min(maxes-mins) == 0:
                import pdb; pdb.set_trace()
            return KDTree.leafnode(idx)
        else:
            data = self.data[idx]
            dmaxes = np.amax(data,axis=0)
            dmins = np.amin(data,axis=0)
            d = np.argmax(dmaxes-dmins)
            maxval = maxes[d]
            minval = mins[d]
            if maxval == minval:
                # all points are identical; warn user?
                return KDTree.leafnode(idx)
            data = data[:,d]

            # sliding midpoint rule; see Maneewongvatana and Mount 1999
            # for arguments that this is a good idea.
            split = np.median(data)  #(maxval+minval)/2
            less_idx = np.nonzero(data <= split)[0]
            greater_idx = np.nonzero(data > split)[0]
            if len(less_idx) == 0:
                split = np.amin(data)
                less_idx = np.nonzero(data <= split)[0]
                greater_idx = np.nonzero(data > split)[0]
            if len(greater_idx) == 0:
                split = np.amax(data)
                less_idx = np.nonzero(data < split)[0]
                greater_idx = np.nonzero(data >= split)[0]
            if len(less_idx) == 0:
                # _still_ zero? all must have the same value
                if not np.all(data == data[0]):
                    raise ValueError("Troublesome data array: %s" % data)
                split = data[0]
                less_idx = np.arange(len(data)-1)
                greater_idx = np.array([len(data)-1])

            lessmaxes = np.copy(maxes)
            lessmaxes[d] = split
            greatermins = np.copy(mins)
            greatermins[d] = split
            
            return KDTree.innernode(d, split,
                    self.__build(idx[less_idx],lessmaxes,mins),
                    self.__build(idx[greater_idx],maxes,greatermins))


def clusters_from_tree(tree, min_bounds, max_bounds):
    try:
        return [tree.idx], [(min_bounds, max_bounds)], ()
    except:
        idx = tree.split_dim
        v = float(tree.split)
        left_max = max_bounds.copy()
        left_max[idx] = v
        right_min = min_bounds.copy()
        right_min[idx] = v
        left_idx, left_bounds, left_splits = clusters_from_tree(tree.less, min_bounds, left_max)
        right_idx, right_bounds, right_splits = clusters_from_tree(tree.greater, right_min, max_bounds)
        
        cluster_idx = left_idx + right_idx
        cluster_bounds = left_bounds + right_bounds
        splits = ((tree.split_dim, tree.split), left_splits, right_splits)
        return cluster_idx, cluster_bounds, splits

def clusters_from_splits(X, idxs, splits):
    try:
        (split_dim, split), left_splits, right_splits = splits
    except:
        return [idxs]
    left_idxs = idxs[X[idxs, split_dim] <= split]
    right_idxs = idxs[X[idxs, split_dim] > split]
    left_clusters = clusters_from_splits(X, left_idxs, left_splits)
    right_clusters = clusters_from_splits(X, right_idxs, right_splits)
    return left_clusters + right_clusters


def plot_tree(tree, min_bounds, max_bounds, hm):
    try:
        tree.idx
    except:
        idx = tree.split_dim
        v = float(tree.split)
        left_max = max_bounds.copy()
        left_max[idx] = v
        right_min = min_bounds.copy()
        right_min[idx] = v
        plot_tree(tree.less, min_bounds, left_max, hm)
        plot_tree(tree.greater, right_min, max_bounds, hm)
        
        if idx == 0: # split on lon, draw vertical line
            hm.bmap.plot([v, v], [min_bounds[1], max_bounds[1]], c="blue")
        else:
            hm.bmap.plot([min_bounds[0], max_bounds[0]], [v, v], c="blue")
            
def tree_neighbors(cluster_bounds):
    neighbors = []
    for i, (min_bounds1, max_bounds1) in enumerate(cluster_bounds):
        minlon1, minlat1 = min_bounds1
        maxlon1, maxlat1 = max_bounds1
        for j, (min_bounds2, max_bounds2) in enumerate(cluster_bounds[:i]):
            minlon2, minlat2 = min_bounds2
            maxlon2, maxlat2 = max_bounds2
            if (minlon1 == maxlon2) or (maxlon1 == minlon2):
                if minlat1 <= maxlat2 and maxlat1 >= minlat2: 
                    neighbors.append((i,j))
            elif (minlat1 == maxlat2) or (maxlat1 == minlat2):
                if minlon1 <= maxlon2 and maxlon1 >= minlon2:
                    neighbors.append((i,j))
            elif (minlon1 == -18 and maxlon2 == 342) or (minlon2 == -18 and maxlon1 == 342):
                if minlat1 <= maxlat2 and maxlat1 >= minlat2: 
                    print "wraparound", i, j
                    neighbors.append((i,j))
    return neighbors

def kdtree_cluster(X, blocksize=300):
    X = X[:, :2].copy()
    lons = X[:, 0] 
    X[:, 0]  = (lons + 18) % 360 - 18

    kdt = KDTree(X, leafsize=blocksize)
    max_bounds = np.array((342, 90.0,))
    min_bounds = np.array((-18.0, -90.0,))
    cluster_idx, cluster_bounds, splits = clusters_from_tree(kdt.tree, min_bounds, max_bounds)
    return cluster_idx, cluster_bounds, splits

