import os
import cPickle as pickle

def load_events(sta="mkar", basedir="/home/dmoore/p_waves/"):
    s = []
    for i in range(1, 1000):
        try:
            with open(os.path.join(basedir, "%s_stuff_%d" % (sta, i * 1000)), 'rb') as f:
                ss = pickle.load(f)
                s += ss
            print "loaded", i
        except IOError:
            with open(os.path.join(basedir,"%s_stuff_final" % (sta,)), 'rb') as f:
                ss = pickle.load(f)
                s += ss
            print "loaded final"
            break

    return s
