import numpy as np
import os

X = []
Y = []
data = []
for i in range(5000):
    try:
        X.append(np.load("clusters/cluster_%03d_X.npy" % i))
    except IOError:
        continue
    Y.append(np.load("clusters/cluster_%03d_Y.npy" % i))
    data.append(np.load("clusters/cluster_%03d_Data.npy" % i))
    print "loaded", i


X = np.vstack(X)
Y = np.vstack(Y)
data = np.vstack(data)

np.save("clusters/aligned_data.npy", data)
np.save("clusters/aligned_X.npy", X)
np.save("clusters/aligned_Y.npy", Y)
