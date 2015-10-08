import numpy as np
import sys

class PDTree(object):
    
    class node(object):
        pass
    
    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(idx)

    class innernode(node):
        def __init__(self, split_vec, center, split, left, right):
            self.split_vec = split_vec
            self.center = center
            self.split = split
            self.left = left
            self.right = right
            self.children = left.children+right.children
            
    def __init__(self, X, minsize):
        self.X = X
        idx = np.arange(len(X))
        self.tree = self.__build(idx, minsize)

        
    def __build(self, idx, minsize):
        n = len(idx)
        if n < minsize:
            return PDTree.leafnode(idx)
        
        
        data = self.X[idx]
        dmean = np.mean(data, axis=0)
        data -= dmean
        XXt = np.dot(data.T, data)
        ev, evec = np.linalg.eig(XXt)
        pidx = np.argmax(ev)
        pvec = evec[:, pidx]
        
        a = np.dot(data, pvec)
        split = np.median(a)
        
        idx1 = idx[a < split]
        idx2 = idx[a >= split]
        
        return PDTree.innernode(pvec, dmean, split, 
                    self.__build(idx1,minsize),
                    self.__build(idx2,minsize))

    def leaf_idx(self):
        
        def child_idxs(node):
            if isinstance(node, PDTree.leafnode):
                return [node.idx]
            else:
                leftidx = child_idxs(node.left)
                rightidx = child_idxs(node.right)
                return leftidx+rightidx
            
        return child_idxs(self.tree)
    
    def recluster(self, X):
        def recluster_recursive(node, idx):
            if isinstance(node, PDTree.leafnode):
                return [idx]
        
            data = X[idx]
            a = np.dot(data - node.center, node.split_vec)
            idx1 = idx[a < node.split]
            idx2 = idx[a >= node.split]
            return recluster_recursive(node.left, idx1) + recluster_recursive(node.right, idx2)
        
        idx = np.arange(len(X))
        return recluster_recursive(self.tree, idx)

def pdtree_cluster(X, blocksize=300):
    X = X[:, :2].copy()
    lons = X[:, 0] 
    X[:, 0]  = (lons + 22) % 360 - 22

    t = PDTree(X, minsize=blocksize)
    idxs = t.leaf_idx()

    def reblock(XX):
        lons = XX[:, 0].copy()
        XX[:, 0]  = (lons + 22) % 360 - 22
        cluster_idxs = t.recluster(XX[:, :2])
        XX[:, 0]  = lons
        return cluster_idxs

    return idxs, reblock

