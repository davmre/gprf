from gprf import GPRF, Blocker
from gprfopt import SampledData, exp_dir, grid_centers

from treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian
from treegp.util import mkdir_p
import numpy as np
import scipy.stats
import scipy.optimize
import time
import os
import sys

import cPickle as pickle

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from collections import defaultdict

RESULT_COLS = {'step': 0, 'time': 1, 'mll': 2, 'dlscale': 3, 'mad': 4,
               'xprior': 5, 'smse_local': 6, 'smse': 7, 'msll_local_block': 8,
               'msll_block': 9, 'msll_local_diag': 10, 'msll_diag': 11}

def plot_ll(run_name):
    steps, times, lls = load_log(run_name)

def load_results(d):
    r = os.path.join(d, "results.txt")

    results = []
    with open(r, 'r') as rf:
        for line in rf:
            try:
                lr = [float(x) for x in line.split(' ')]
                results.append(lr)
            except:
                continue
    return np.asarray(results)

def read_result_line(s):
    r = {}
    parts = s.split(' ')
    for lbl, col in RESULT_COLS.items():
        p = parts[col]
        if p=="trueX": continue
        try:
            intP = int(p)
            r[lbl] = intP
        except:
            floatP = float(p)
            r[lbl] = floatP
    return r

def load_final_results(d):
    r = os.path.join(d, "results.txt")

    results = []
    with open(r, 'r') as rf:
        lines = rf.readlines()
        r_final = read_result_line(lines[-2])
        r_true = read_result_line(lines[-1])
    return r_final, r_true


def vis_points(run=None, d=None, sdata_file=None, y_target=0, seed=None, blocksize=None, highlight_block=None):

    if d is None:
        d = exp_dir(run)

    if sdata_file is not None:
        with open(sdata_file, 'rb') as f:
            sdata = pickle.load(f)

    for fname in  ["true.xxx",] + sorted(os.listdir(d)):
        if fname == "true.xxx":
            X = sdata.SX
        elif not fname.startswith("step") or not fname.endswith("_X.npy"): 
            continue
        else:
            X = np.load(os.path.join(d,fname))

        try:
            ix_fname = fname.replace("_X", "_IX")
            IX = np.load(os.path.join(d,ix_fname))
        except:
            IX = None

        fig = Figure(dpi=144, figsize=(14, 14))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)

        cmap = "jet"
        sargs = {}
        if y_target==-1:
            # plot "wrongness"
            c = np.sqrt(np.sum((X - sdata.SX)**2, axis=1))
            cmap="hot"
        elif y_target==-2 or y_target==-3:
            # plot blocks
            c = np.zeros((X.shape[0]))

            if y_target==-2:
                np.random.seed(seed)
                sdata.cluster_rpc(blocksize)
            else:
                centers = grid_centers(blocksize)
                sdata.set_centers(centers)

            cmap ="prism"
            if highlight_block is not None:
                block_colors = np.ones(( len(sdata.block_idxs),)) * 0.4
                block_colors[highlight_block] = 0.0
            else:
                block_colors = np.linspace(0.0, 1.0, len(sdata.block_idxs))
            block_idxs = sdata.reblock(X)
            for i, idxs in enumerate(block_idxs):
                c[idxs] = block_colors[i]

            #c = np.sqrt(np.sum((X - sdata.SX)**2, axis=1))
        elif sdata_file is None:
            c = None
        else:
            c = sdata.SY[:, y_target:y_target+1].flatten()
            sargs['vmin'] = -3.0
            sargs['vmax'] = 3.0

        
        npts = len(X)
        xmax = np.sqrt(npts)
        X *= xmax

        if IX is not None:
            IX *= xmax
            ax.scatter(IX[:, 0], IX[:, 1], alpha=1.0, c="black", s=25, marker='o', linewidths=0.0, **sargs)


        ax.scatter(X[:, 0], X[:, 1], alpha=1.0, c=c, cmap=cmap, s=70, marker='.', linewidths=0.0, **sargs)
        ax.set_xlim((0,xmax))
        ax.set_ylim((0,xmax))

        ax.set_yticks([20, 40, 60, 80, 100])

        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)

        canvas = FigureCanvasAgg(fig)

        out_name = os.path.join(d, fname[:-4] + ".png")
        fig.savefig(out_name, bbox_inches="tight")
        print "wrote", out_name

    print "generating movie...:"
    cmd = "avconv -f image2 -r 5 -i step_%05d_X.png -qscale 28 gprf.mp4".split(" ")
    import subprocess
    p = subprocess.Popen(cmd, cwd=d)
    p.wait()
    print "done"


def write_plot(plot_data, out_fname, xlabel="Time (s)",
               ylabel="", ylim=None, xlim=None, plot_args = None):

    fig = Figure(dpi=144)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)

    if plot_args is None:
        plot_args = lambda x : dict()

    for label, (x, y) in sorted(plot_data.items()):
        ax.plot(x, y, label=label, **plot_args(label))

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.legend()



    canvas = FigureCanvasAgg(fig)
    #can.print_figure('test')
    fig.savefig(out_fname)

def eighty_run_params():
    yd = 50
    seed = 0
    method = "l-bfgs-b"
    ntest  = 500

    ntrain = 80000
    local_nblocks = [16, 36, 100, 196, 400, 900]
    gprf_nblocks = [100, 196, 400, 900]

    runs = []

    runs_by_key = defaultdict(list)

    runs_gprf = []
    runs_local = []


    lscale = 6.0 / np.sqrt(ntrain)
    obs_std = 2.0 / np.sqrt(ntrain)

    for nblocks in local_nblocks:
        run_params_local = {'n': ntrain, 'ntest': ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 1.0, "method": method, 'nblocks': nblocks, 'task': 'x', 'noise_var': 0.01, "num_inducing": 0} 
        runs_local.append(run_params_local)

        key = "Local-%d" % nblocks
        runs_by_key[key].append(run_params_local)

    for nblocks in gprf_nblocks:
        run_params_gprf = {'n': ntrain, 'ntest': ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.1, "method": method, 'nblocks': nblocks, 'task': 'x', 'noise_var': 0.01, "num_inducing": 0} 
        runs_gprf.append(run_params_gprf)

        key = "GPRF-%d" % nblocks
        runs_by_key[key].append(run_params_gprf)


    runs += runs_local
    runs += runs_gprf



    return runs, runs_by_key

def truegp_run_params():
    yd = 50
    seed = 0
    method = "l-bfgs-b"
    ntest  = 500

    local_nblocks = [1, 9, 25, 49, 100]
    gprf_nblocks = [9, 25, 49, 100]
    ns_inducing = [200, 500, 1000, 2000, ]
    runs = []
    runs_by_key = defaultdict(list)
    ntrain = 10000

    runs_gprf = []
    runs_local = []
    runs_fitc = []

    lscale = 6.0 / np.sqrt(ntrain)
    obs_std = 2.0 / np.sqrt(ntrain)

    init_true = False

    for nblocks in local_nblocks:
        run_params_local = {'n': ntrain, 'n': ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 1.0, "method": method, 'nblocks': nblocks, 'task': 'x', 'noise_var': 0.01, "num_inducing": 0, "init_true": init_true} 
        runs_local.append(run_params_local)

        key = "Local-%d" % nblocks
        runs_by_key[key].append(run_params_local)

    for nblocks in gprf_nblocks:
        run_params_gprf = {'n': ntrain, 'ntest': ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.1, "method": method, 'nblocks': nblocks, 'task': 'x', 'noise_var': 0.01, "num_inducing": 0, "init_true": init_true} 
        runs_gprf.append(run_params_gprf)

        key = "GPRF-%d" % nblocks
        runs_by_key[key].append(run_params_gprf)

    for num_inducing in ns_inducing:
        run_params_inducing = {'n': ntrain, 'ntest': ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed,  "method": method,  'task': 'x', 'noise_var': 0.01, 'gplvm_type': "sparse", 'num_inducing': num_inducing, "nblocks": 1, "local_dist": 1.0, "init_true": init_true}
        runs_fitc.append(run_params_inducing)
        key = "FITC-%d" % num_inducing
        runs_by_key[key].append(run_params_inducing)

    runs += runs_local
    runs += runs_gprf
    runs += runs_fitc


    return runs, runs_by_key

def fitc_run_params(obs_std_base=2.0):
    yd = 50
    seed = 0
    method = "l-bfgs-b"
    ntest  = 500

    #ntrains = [5000, 10000, 15000, 20000]
    ntrains = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000]

    ns_inducing = [200, 500, 1000, 2000, ]

    def square_up(n):
        return int(np.ceil(np.sqrt(n)))
    def square_down(n):
        return int(np.floor(np.sqrt(n)))    
    def get_nblocks(ntrain, block_size_target):
        return square_down(ntrain / float(block_size_target))**2

    local_block_size = [200,400]
    gprf_block_size = [200,400]

    runs = []

    runs_by_key = defaultdict(list)

    for ntrain in ntrains:
        runs_gprf = []
        runs_local = []
        runs_fitc = []

        #lscale = 5.4772255750516621 / np.sqrt(ntrain)
        #obs_std = 1.0954451150103324 / np.sqrt(ntrain)
        lscale = 6.0 / np.sqrt(ntrain)
        obs_std = obs_std_base / np.sqrt(ntrain)

        for blocksize in local_block_size:
            nblocks = get_nblocks(ntrain, blocksize)
            actual_blocksize = ntrain / float(nblocks)
            if actual_blocksize >= 8000: continue
            print ntrain, "target", blocksize, "actual", actual_blocksize
            run_params_local = {'n': ntrain, 'ntest': ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 1.0, "method": method, 'nblocks': nblocks, 'task': 'xcov', 'noise_var': 0.01, "num_inducing": 0} 
            runs_local.append(run_params_local)

            key = "Local-%d" % blocksize
            runs_by_key[key].append(run_params_local)

        for blocksize in gprf_block_size:
            nblocks = get_nblocks(ntrain, blocksize)
            run_params_gprf = {'n': ntrain, 'ntest': ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.1, "method": method, 'nblocks': nblocks, 'task': 'xcov', 'noise_var': 0.01, "num_inducing": 0} 
            runs_gprf.append(run_params_gprf)

            key = "GPRF-%d" % blocksize
            runs_by_key[key].append(run_params_gprf)

        for num_inducing in ns_inducing:
            if num_inducing >= ntrain: continue

            run_params_inducing = {'n': ntrain, 'ntest': ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed,  "method": method,  'task': 'xcov', 'noise_var': 0.01, 'gplvm_type': "sparse", 'num_inducing': num_inducing, "nblocks": 1, "local_dist": 1.0}
            runs_fitc.append(run_params_inducing)
            key = "FITC-%d" % num_inducing
            runs_by_key[key].append(run_params_inducing)

        runs += runs_local
        runs += runs_gprf
        runs += runs_fitc


    return runs, runs_by_key




def gen_runexp(runs, base_cmd, outfile, tail="", analyze=False, parallel=True, maxsec=5400):

    f_out = open(outfile, 'w')

    for run in runs:
        args = ["--%s=%s" % (k,v) for (k,v) in sorted(run.items(), key=lambda x: x[0]) if k!= "init_true"]
        
        if analyze:
            args.append("--analyze")
            args.append("--analyze_full")
        if parallel:
            args.append("--parallel")
        if "init_true" in run and  run["init_true"]:
            args.append("--init_true")
        if 'maxsec' not in run and maxsec is not None:
            args.append("--maxsec=%d" % maxsec)
        cmd = base_cmd + " " + " ".join(args)
        f_out.write(cmd + tail + "\n")

    f_out.close()

def gen_runs():
    cloud_base = "sudo su -c \"bash /home/sigvisa/python/gprf/run_cloud.sh gprfopt.py"
    cloud_base_limited = "sudo su -c \"bash /home/sigvisa/python/gprf/run_cloud_limit.sh gprfopt.py"
    cloud_tail = "\" sigvisa"

    standard_base = "python gprfopt.py"
    standard_tail = ""

    runs_eighty, _ = eighty_run_params()
    runs_truegp, _ = truegp_run_params()
    runs_fitc, _ = fitc_run_params()

    gen_runexp(runs_eighty, standard_base, "run_eighty.sh", analyze=False, maxsec=86400, parallel=False, tail=standard_tail)
    gen_runexp(runs_truegp, standard_base, "run_truegp.sh", analyze=False, maxsec=18000, parallel=False, tail=standard_tail)
    gen_runexp(runs_fitc, standard_base, "run_fitc.sh", analyze=False, maxsec=36000, parallel=False, tail=standard_tail)



def main():
    if len(sys.argv) > 1 and sys.argv[1] =="vis":
        y_target = -1
        seed = None
        blocksize = None
        highlight_block = None
        if len(sys.argv) > 4:
            y_target = int(sys.argv[4])
            if len(sys.argv) > 5:
                seed = int(sys.argv[5])
                blocksize = int(sys.argv[6])
            if len(sys.argv) > 7:
                highlight_block = int(sys.argv[7])
        vis_points(d=sys.argv[2], y_target=y_target, sdata_file=sys.argv[3], seed=seed, blocksize=blocksize, highlight_block=highlight_block)
    else:
        gen_runs()

if __name__ =="__main__":
    main()
