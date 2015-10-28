import os
import numpy as np
from seismic.seismic_util import scraped_to_evid_dict
from run_seismic import dist_lld
from treegp.gp import sort_morton

COL_TIME, COL_TIMEERR, COL_LON, COL_LAT, COL_SMAJ, COL_SMIN, COL_STRIKE, COL_DEPTH, COL_DEPTHERR = np.arange(9)

homedir = os.getenv("HOME")
isc_dict = scraped_to_evid_dict(os.path.join(homedir, "python/gprf/isc.txt"))
idc_dict = scraped_to_evid_dict(os.path.join(homedir, "python/gprf/idc.txt"))

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


