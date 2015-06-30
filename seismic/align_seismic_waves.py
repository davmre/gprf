import time
from sigvisa.utils.geog import dist_km
import numpy as np

def xcorr_valid(a,b):
    a = (a - np.mean(a)) / (np.std(a) * np.sqrt(len(a)))
    b = (b - np.mean(b)) / (np.std(b) * np.sqrt(len(a)))

    xc = my_xc(a, b)
    xcmax = np.max(xc)
    offset = np.argmax(xc)
    return xcmax, offset, xc

import scipy.weave as weave
from scipy.weave import converters
def my_xc(a, b):
    # assume len(a) < len(b)
    n = len(b) - len(a)+1
    m = len(a)
    r = np.zeros((n,))
    a_normed = a / np.linalg.norm(a)
    code="""
for(int i=0; i < n; ++i) {
    double b_norm = 0;
    double cc = 0;
    for (int j=0; j < m; ++j) {
        cc += a_normed(j)*b(i+j);
        b_norm += b(i+j)*b(i+j);
    }
    cc /= sqrt(b_norm);
    r(i) = cc;
}
"""
    weave.inline(code,['n', 'm', 'a_normed', 'b', 'r',],type_converters = converters.blitz,verbose=2,compiler='gcc')
    """
    for i in range(n):
        window = b[i:i+len(a)]
        w_normed = window / np.linalg.norm(window)
        r[i] = np.dot(a_normed, w_normed)
    """
    return r



import cPickle as pickle

def load_events(sta="mkar"):
    s = []
    for i in range(1, 100):
        try:
            with open("/home/dmoore/p_waves/%s_stuff_%d" % (sta, i * 1000), 'rb') as f:
                ss = pickle.load(f)
                s += ss
            print "loaded", i
        except IOError:
            with open("/home/dmoore/p_waves/%s_stuff_final" % (sta,), 'rb') as f:
                ss = pickle.load(f)
                s += ss
            print "loaded final"
            break

    return s

window_start_idx = 60 # 2s before IDC arrival
window_end_idx = 260 # 8s after IDC arrival (so, 10s window)
t = np.linspace(-3.0, 10.0, 301)
prior_std = -np.abs(t)/3.0

def align(w1, w2):
    patch1 = w1[window_start_idx:window_end_idx]
    patch2 = w2[window_start_idx:window_end_idx]

    xc1 = my_xc(patch1, w2)
    xc2 = my_xc(patch2, w1)

    align1 = np.argmax(xc1 + prior_std)
    align2 = np.argmax(xc2 + prior_std)
    xcmax1 = xc1[align1]
    xcmax2 = xc2[align2]

    # adj1 means, how much would I have to shift the window start time of w1 so that it lines up perfectly with w2
    # (in its current form).
    # align1 = windowstart implies that the patch from w1 lines up with windowstart in w2, so this is the zero point.
    # align1 = windowstart+1 implies that the patch from w1 lines up with windowstart+1 in w2.
    # We can make them align by "slipping w1 to the right", so that the patch includes an extra early index.
    # or we can equivalent "pull the patch to the left", which is what I want to do.
    adj1 = window_start_idx - align1
    adj2 = window_start_idx - align2

    return xcmax1, xcmax2, align1, align2, adj1, adj2

COL_IDX, COL_EVID, COL_LON, COL_LAT, COL_SMAJ, COL_SMIN, COL_STRIKE, COL_DEPTH, COL_DEPTHERR = np.arange(9)
def load_seismic_locations():
    fname = "/home/dmoore/python/sigvisa/scraped.txt"
    return np.loadtxt(fname, delimiter=",")

def extract_patches(waves, window_starts):
    patch_len = 200
    patches = []
    for w, ws in zip(waves, window_starts):
        start_idx = int(ws)
        patch = w[start_idx:start_idx+patch_len].copy()
        patch -= np.mean(patch)
        patch /= np.linalg.norm(patch)
        patches.append(patch)
    return patches

def cluster_waves(fds):
    n = fds.shape[0]
    srate = 20.0
    waves = []
    for row in fds:
        idx = int(row[COL_IDX])
        w = s[idx][1][0]
        waves.append(w)
    return waves

def offsets(ws):
    n = len(ws)
    xcmax1 = np.zeros((n,n))
    xcmax2 = np.zeros((n,n))
    offset1 = np.zeros((n,n))
    offset2 = np.zeros((n,n))
    adj1 = np.zeros((n,n))
    adj2 = np.zeros((n,n))
    for i, w1 in enumerate(ws):
        for j, w2 in enumerate(ws[:i]):
              xcmax1[i,j], xcmax2[i,j], offset1[i,j], offset2[i,j], adj1[i,j], adj2[i,j] = align(w1, w2)
    return xcmax1, xcmax2, offset1, offset2, adj1, adj2

def correlate_patches(patches):
    p = np.array(patches)
    P = np.dot(p, p.T)
    P -= np.diag(np.diag(P))
    return P

def correlation_surface(waves, window_idxs, i, xcmax=None, threshold=0.45):
    patches = extract_patches(waves, window_idxs)
    w = waves[i]
    total_xc = np.zeros(301)
    for j, patch in enumerate(patches):
        if j==i: continue
        if xcmax[i,j] > threshold:
            xc = my_xc(patch, w)
            total_xc += xc * xcmax[i,j]
    return total_xc

def coherency(waves, window_idxs):
    patches = extract_patches(waves, window_idxs)
    P = correlate_patches(patches)
    coherency = np.mean(P)
    return coherency

def distances(fds):
    n = len(fds)
    ds = np.zeros((n,n))
    for i, row1 in enumerate(fds):
        for j, row2 in enumerate(fds[:i]):
            ll1 = (row1[COL_LON], row1[COL_LAT])
            ll2 =  (row2[COL_LON], row2[COL_LAT])
            ds[i,j] = dist_km(ll1, ll2)
    ds += ds.T
    return ds


s = load_events()
t = np.linspace(-3.0, 10.0, 301)
prior = -np.abs(t)/1.0

def coordinate_ascent(waves, window_idxs, xcmax, threshold=0.4):
    #c0 = coherency(waves, window_idxs)
    perm = np.random.permutation(len(waves))
    for i in perm:
        surface = correlation_surface(waves, window_idxs, i, xcmax=xcmax, threshold=threshold)
        window_idxs[i] = np.argmax(surface + prior)
        #c1 = coherency(waves, window_idxs)
        #print c0, c1
        #c0 = c1
    return window_idxs

def align_waves(waves, nruns = 5, threshold=0.45, max_s=None, init_widxs=None):
    xcmax1, xcmax2, offset1, offset2, adj1, adj2 = offsets(waves)
    xcmax = np.max((xcmax1, xcmax2),axis=0)
    xcmax += xcmax.T

    n = len(waves)

    def coord_ascent_run():
        window_idxs = np.ones((n), dtype=float)*(85+np.random.randn()*3) + np.random.randn(n) * 5
        window_idxs = coordinate_ascent(waves, window_idxs, xcmax, threshold)
        window_idxs = coordinate_ascent(waves, window_idxs, xcmax, threshold)
        window_idxs = coordinate_ascent(waves, window_idxs, xcmax, threshold)
        window_idxs = coordinate_ascent(waves, window_idxs, xcmax, threshold)
        c = coherency(waves, window_idxs)
        return window_idxs, c

    best_c = 0.0
    best_widxs = init_widxs
    if best_widxs is not None:
        best_c = coherency(waves, best_widxs)

    t0 = time.time()
    for i in range(nruns):
        widxs, c = coord_ascent_run()
        print c
        if c > best_c:
            best_c = c
            best_widxs = widxs
        if max_s is not None:
            t1 = time.time()
            if t1-t0 > max_s:
                break

    return best_c, best_widxs

fd = load_seismic_locations()
lls = fd[:, [COL_LON, COL_LAT]]




from sklearn.cluster import KMeans
np.random.seed(0)
km = KMeans(n_clusters=500, init='k-means++', n_init=2, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=1, random_state=None, copy_x=True, n_jobs=1)
r = km.fit(lls)
clusters = []
for i in range(500):
    idx = km.labels_==i
    lli =lls[idx,:]
    clusters.append(fd[idx, :])
    #c = np.mean(np.std(lli, axis=0))
    #if len(lli) > 50 and c < .25:
    #    print i, len(lli), c
    #    clusters.append(fd[idx, :])

for i, cluster in enumerate(clusters):
    try:
        waves = cluster_waves(cluster)
        c, widxs = align_waves(waves, nruns=30, threshold=0.4, max_s=15)
        patches = extract_patches(waves, widxs)
        YS = np.array(patches)
        XS = np.array(cluster[:, (COL_LON, COL_LAT, COL_DEPTH)])
        np.save("clusters/cluster_%03d_Data.npy" % i, cluster)
        np.save("clusters/cluster_%03d_X.npy" % i, XS)
        np.save("clusters/cluster_%03d_Y.npy" % i, YS)
        print "saved cluster", i
    except Exception as e:
        continue
