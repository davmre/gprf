from gprf import GPRF, Blocker
from gprfopt import SampledData, exp_dir, dump_covs, grid_centers

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



RESULT_COLS = {'step': 0, 'time': 1, 'mll': 2, 'dlscale': 3, 'mad': 4,
               'xprior': 5, 'predll': 6, 'predll_neighbors': 7}

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

def load_plot_data(runs, target="predll", running_best=True):

    col = RESULT_COLS[target]

    plot_data = {}
    for label, run_params in runs.items():
        results = load_results(exp_dir(run_params))
        t = results[:, 1]
        y = results[:, col]

        if running_best:
            lls = results[:, 2]
            best_lls = [np.argmax(lls[:i+1]) for i in range(len(lls))]
            y = y[best_lls]

        plot_data[label] = (t, y)

    return plot_data

def vis_points(run=None, d=None, sdata_file=None, y_target=0, seed=None, blocksize=None, highlight_block=None):

    if d is None:
        d = exp_dir(run)

    if sdata_file is not None:
        with open(sdata_file, 'rb') as f:
            sdata = pickle.load(f)

    for fname in sorted(os.listdir(d)):
        if not fname.startswith("step") or not fname.endswith("_X.npy"): continue
        X = np.load(os.path.join(d,fname))

        fig = Figure(dpi=144)
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
        ax.scatter(X[:, 0], X[:, 1], alpha=0.4, c=c, cmap=cmap, s=25, marker='.', linewidths=0.0, **sargs)
        ax.set_xlim((0,1))
        ax.set_ylim((0,1))

        canvas = FigureCanvasAgg(fig)

        out_name = os.path.join(d, fname[:-4] + ".png")
        fig.savefig(out_name)
        print "wrote", out_name

    print "generating movie...:"
    cmd = "avconv -f image2 -r 5 -i step_%05d_X.png -qscale 28 gprf.mp4".split(" ")
    import subprocess
    p = subprocess.Popen(cmd, cwd=d)
    p.wait()
    print "copying to ~/public_html/gprf.mp4"
    p = subprocess.Popen(["cp", "gprf.mp4", "/home/dmoore/public_html/"], cwd=d)
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


def fixedsize_run_params(lscale=0.4, obs_std=0.1, rpc_cluster=False):
    ntrain = 15000
    n = 15550
    yd = 50
    seed=0
    method="l-bfgs-b"
    local_dist=0.4

    base_params = {'ntrain': ntrain, 'n': n, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': local_dist, "method": method, 'task': 'x', 'noise_var': 0.01}

    runs = {'GP': base_params}
    block_counts = [9, 16, 25, 36, 49, 64]
    cluster_sizes = [5000, 2000, 1000, 500, 300, 100]

    if rpc_cluster:
        for bs in cluster_sizes:
            rfp = base_params.copy()
            rfp['rpc_blocksize'] = bs
            runs['GPRF-%d' % bs] = rfp

            localp = rfp.copy()
            localp['local_dist'] = 1.0
            runs['Local-%d' % bs] = localp
    else:
        for bc in block_counts:
            rfp = base_params.copy()
            rfp['nblocks'] = bc
            runs['GPRF-b%d' % bc] = rfp

            localp = rfp.copy()
            localp['local_dist'] = 1.0
            runs['Local-b%d' % bc] = localp
    return runs




def plot_models_fixedsize(**kwargs):
    runs = fixedsize_run_params(**kwargs)
    ylims = {'predll': (-3, 0),
             'predll_neighbors': (-3, 0),
             'mad': (0.0, 0.1)}


    def plot_args(label):
        args = {}
        colors = {1: "black", 4: "red", 9: "green", 16: "blue", 25: "purple", 36: "orange", 49: "brown"}
        if "Local" in label:
            args['linestyle'] = '--'
            args['color'] = colors[np.abs(int(label[-2:]))]
        elif "GPRF" in label:
            args['linestyle'] = '-'
            args['color'] = colors[np.abs(int(label[-2:]))]
        elif label==GP:
            args['linestyle'] = '.'
        return args

    for target in ("predll", "predll_neighbors", "mad"):
        plot_data = load_plot_data(runs, target=target)
        write_plot(plot_data, out_fname="fixedsize_%s.png" % target,
                   ylabel=target, ylim=ylims[target], plot_args=plot_args, xlim=(0, 1800))

def growing_run_params():
    yd = 50
    seed = 4
    method = "l-bfgs-b"
    ntest  =500

    ntrains = [500, 2000, 4500, 8000, 12500, 18000, 24500]
    nblocks = [1, 4, 9, 16, 25, 36, 49]
    lscales = [1.0, 0.66666, 0.5, 0.4, 0.333333, 0.2856]
    obs_stds = [0.2, 0.1333333, 0.1, 0.08, 0.0666666, 0.05714]

    x = ntrains
    runs_gprf = []
    runs_local = []
    runs_full = []

    for ntrain, nblock, lscale, obs_std in zip(ntrains, nblocks, lscales, obs_stds):
        run_params_gprf = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.05, "method": method, 'nblocks': nblock, 'task': 'x', 'noise_var': 0.01}
        runs_gprf.append(run_params_gprf)

        run_params_local = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.00, "method": method, 'nblocks': nblock, 'task': 'x', 'noise_var': 0.01}
        runs_local.append(run_params_local)

        run_params_full = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.00, "method": method, 'nblocks': 1, 'task': 'x', 'noise_var': 0.01}
        runs_full.append(run_params_full)

    return runs_gprf, runs_local, runs_full


def seismic_run_params():

    #npts = [1000, 5000, 10000, 13000]

    npts = [500, 3000, 5000, 10000, 19980]

    rpc_sizes = [50, 200, 800]
    thresholds = [1.0, 0.001]

    runs_full = []
    runs_gprf = []
    init_cov = "seismic_experiments/19980_200_1.0000_default_cov/step_00030_cov.npy"
    for n in npts:
        if n <= 5000:
            run_params_full = {'n': n, 'threshold': 0.00, 'task': 'x', 'rpc_blocksize': -1, 'init_cov': init_cov, 'maxsec': 10000}
            runs_full.append(run_params_full)


        for rpc_size in rpc_sizes:
            if rpc_size > n:
                continue
            for threshold in thresholds:
                run_params_gprf = {'n': n, 'threshold': threshold, 'task': 'x', 'rpc_blocksize': rpc_size, 'init_cov': init_cov}
                run_params_gprf['maxsec'] = n
                runs_gprf.append(run_params_gprf)

    return runs_gprf+runs_full


def fault_run_params():
    yd = 50
    seed = 1004
    method = "l-bfgs-b"
    ntest  =500

    ntrains = [1000, 3000, 5000, 10000, 20000]
    rpc_sizes = [200, 500]
    lscales = [0.1, 0.05]
    obs_stds = [0.02,]
    local_dists = [0.0, 0.001, 0.01]

    x = ntrains
    runs_gprf = []
    runs_local = []
    runs_full = []

    for ntrain in ntrains:
        for lscale in lscales:
            for obs_std in obs_stds:
                run_params_full = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.05, "method": method, 'nblocks': 1, 'task': 'x', 'noise_var': 0.01}
                runs_full.append(run_params_full)

                for rpc_size in rpc_sizes:
                    for local_dist in local_dists:
                        run_params_gprf = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': local_dist, "method": method, 'rpc_blocksize': rpc_size, 'task': 'x', 'noise_var': 0.01}
                        runs_gprf.append(run_params_gprf)

    return runs_gprf+runs_full


def crazylines_run_params():
    yd = 50
    seed = 1355
    method = "l-bfgs-b"
    ntest  = 500

    #ntrains = [5000, 10000, 15000, 20000]
    ntrains = [20000, 30000, 40000, 60000, 80000]
    rpc_sizes = [200, 1000]
    local_dists = [1.0, 0.01]

    x = ntrains
    runs_gprf = []
    runs_local = []
    runs_full = []

    for ntrain in ntrains:
        #lscale = 5.4772255750516621 / np.sqrt(ntrain)
        #obs_std = 1.0954451150103324 / np.sqrt(ntrain)
        lscale = 1.5 / np.sqrt(ntrain)
        obs_std = 0.3 / np.sqrt(ntrain)


        run_params_full = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': 0.05, "method": method, 'nblocks': 1, 'task': 'x', 'noise_var': 0.01}
        runs_full.append(run_params_full)

        for rpc_size in rpc_sizes:
            for local_dist in local_dists:
                ld = 1.0 if local_dist==1.0 else lscale*2
                run_params_gprf = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed, 'local_dist': ld, "method": method, 'rpc_blocksize': rpc_size, 'task': 'x', 'noise_var': 0.01}
                runs_gprf.append(run_params_gprf)

    return runs_gprf+runs_full

def crazylines_run_gpy_params():
    yd = 50
    seed = 1355
    method = "l-bfgs-b"
    ntest  = 500

    #ntrains = [5000, 10000, 15000, 20000]
    ntrains = [20000, 30000, 40000, ]
    methods = ["sparse", ]
    ns_inducing = [1000, 2000, 4000, 10000]
    #rpc_sizes = [200, 1000]
    #local_dists = [1.0, 0.01]

    x = ntrains
    runs_gprf = []
    runs_local = []
    runs_full = []

    for ntrain in ntrains:
        #lscale = 5.4772255750516621 / np.sqrt(ntrain)
        #obs_std = 1.0954451150103324 / np.sqrt(ntrain)
        lscale = 1.5 / np.sqrt(ntrain)
        obs_std = 0.3 / np.sqrt(ntrain)

        for gplvm_type in methods:
            for num_inducing in ns_inducing:
                if num_inducing > ntrain/4.5: continue
                if gplvm_type=="bayesian" and num_inducing > 400: continue
                
                run_params_gprf = {'ntrain': ntrain, 'n': ntrain+ntest, 'lscale': lscale, 'obs_std': obs_std, 'yd': yd, 'seed': seed,  "method": method,  'task': 'x', 'noise_var': 0.01, 'gplvm_type': gplvm_type, 'num_inducing': num_inducing}
                runs_gprf.append(run_params_gprf)

    return runs_gprf



def plot_models_growing():

    runs_gprf, runs_local, runs_full = growing_run_params()

    times_local = []
    times_gprf = []
    times_full = []
    mad_local = []
    mad_gprf = []
    mad_full = []
    predll_local = []
    predll_gprf = []
    predll_full = []
    predll_true_local = []
    predll_true_gprf = []
    predll_true_full = []
    ntrains = []

    for (gprf, local, full) in zip(runs_gprf, runs_local, runs_full):

        ntrains.append(gprf['ntrain'])

        def process(stuff, times, mads, predlls, predlls_true):
            d = exp_dir(stuff)
            r = load_results(d)
            if len(r) == 0:
                times.append(0)
                mads.append(0)
                predlls.append(0)
                predlls_true.append(0)
                return

            times.append(np.mean(np.diff(r[:, 1])))
            fr, tr = load_final_results(d)
            mads.append(fr['mad'])
            predlls.append(fr['predll'])
            predlls_true.append(tr['predll'])

        process(gprf, times_gprf, mad_gprf, predll_gprf, predll_true_gprf)
        process(local, times_local, mad_local, predll_local, predll_true_local)
        process(full, times_full, mad_full, predll_full, predll_true_full)

    pd_times = {'GPRF': (ntrains, times_gprf),
                'Local': (ntrains, times_local),
                "GP": (ntrains, times_full)}
    pd_mad = {'GPRF': (ntrains, mad_gprf),
                'Local': (ntrains, mad_local),
                "GP": (ntrains, mad_full)}
    pd_predll = {'GPRF': (ntrains, predll_gprf),
                'Local': (ntrains, predll_local),
                 "GP": (ntrains, predll_full)}
    pd_predll_true = {'GPRF': (ntrains, predll_true_gprf),
                'Local': (ntrains, predll_true_local),
                      "GP": (ntrains, predll_true_full)}

    write_plot(pd_times, "times.png", xlabel="n", ylabel="gradient evaluation time (s)")
    write_plot(pd_mad, "mad.png", xlabel="n", ylabel="X locations: mean absolute deviation")
    write_plot(pd_predll, "predll.png", xlabel="n", ylabel="test MSLL")
    write_plot(pd_predll_true, "predll_true.png", xlabel="n", ylabel="test MSLL from true X")

def cov_run_params_hard():

    #lscales = [0.4, 0.1, 0.02]
    lscale = 0.4
    noises = [0.1, 1.0]

    runs = []
    for noise_var in noises:
        for init_seed in range(2):
            run_gprf = {'ntrain': 15000, 'n': 15500, 'lscale': lscale, 'obs_std': 0.001, 'yd': 10, 'seed': 0, 'local_dist': 0.05, "method": 'l-bfgs-b', 'nblocks': 36, 'task': 'cov', 'init_seed': init_seed, 'noise_var': noise_var}
            run_full = run_gprf.copy()
            run_full['nblocks'] = 1

            run_local = run_gprf.copy()
            run_local['local_dist'] = 0.00

            runs.append(run_gprf)
            #runs.append(run_full)
            runs.append(run_local)
    return runs

def xcov_run_params():

    lscales = [0.4, 0.1,]
    noise_var = 0.01
    #lscale = 0.4
    obs_std = 0.02

    runs = []
    for lscale in lscales:
        for init_seed in range(3):
            run_gprf = {'ntrain': 15000, 'n': 15500, 'lscale': lscale, 'obs_std': obs_std, 'yd': 50, 'seed': 0, 'local_dist': 0.05, "method": 'l-bfgs-b', 'nblocks': 49, 'task': 'xcov', 'init_seed': init_seed, 'noise_var': noise_var}
            run_full = run_gprf.copy()
            run_full['nblocks'] = 1

            run_local = run_gprf.copy()
            run_local['local_dist'] = 0.00

            runs.append(run_gprf)
            #runs.append(run_full)
            runs.append(run_local)
    return runs

def cov_run_params():

    lscales = [0.4, 0.1, 0.02]

    runs = []
    for lscale in lscales:
        for init_seed in range(2):
            run_gprf = {'ntrain': 10000, 'n': 10500, 'lscale': lscale, 'obs_std': 0.001, 'yd': 50, 'seed': 0, 'local_dist': 0.05, "method": 'l-bfgs-b', 'nblocks': 25, 'task': 'cov', 'init_seed': init_seed, 'noise_var': 0.01}
            run_full = run_gprf.copy()
            run_full['nblocks'] = 1

            run_local = run_gprf.copy()
            run_local['local_dist'] = 0.00

            runs.append(run_gprf)
            runs.append(run_full)
            runs.append(run_local)
    return runs


def gen_runexp(runs, base_cmd, outfile, analyze=False, parallel=True, maxsec=5400):

    f_out = open(outfile, 'w')

    for run in runs:
        args = ["--%s=%s" % (k,v) for (k,v) in sorted(run.items(), key=lambda x: x[0]) ]
        if analyze:
            args.append("--analyze")
        if parallel:
            args.append("--parallel")
        if 'maxsec' not in run and maxsec is not None:
            args.append("--maxsec=%d" % maxsec)
        cmd = base_cmd + " " + " ".join(args)
        f_out.write(cmd + "\n")

    f_out.close()

def gen_runs():
    #runs_cov = cov_run_params()
    runs_fixedsize = fixedsize_run_params(lscale=0.1, obs_std=0.02, rpc_cluster=True).values()
    gen_runexp(runs_fixedsize, "python gprfopt.py", "run_fixedsize.sh", analyze=False)
    runs_fixedsize = fixedsize_run_params(lscale=0.1, obs_std=0.02).values()
    gen_runexp(runs_fixedsize, "python gprfopt.py", "run_fixedsize_blocks.sh", analyze=False)
    #plot_models_fixedsize(lscale=0.1, obs_std=0.02)

    #runs_growing = np.concatenate(growing_run_params())

    #runs_cov = cov_run_params_hard()
    #runs_xcov = xcov_run_params()

    #runs_fault = fault_run_params()
    runs_lines = crazylines_run_gpy_params()
    gen_runexp(runs_lines, "python gprfopt.py", "run_lines_gpy_big.sh", analyze=False, maxsec=10800)
    runs_lines = crazylines_run_params()
    gen_runexp(runs_lines, "python gprfopt.py", "run_lines_big.sh", analyze=False, maxsec=10800)
    #gen_runexp(runs_lines, "python python/gprf/gprfopt.py", "analyze_lines.sh", analyze=True)

    #runs_seismic = seismic_run_params()
    #gen_runexp(runs_seismic, "python run_seismic.py", "run_seismic.sh", analyze=False, maxsec=14400)
    #gen_runexp(runs_lines, "python python/bcm/treegp/bcm/bcmopt.py", "analyze_lines.sh", analyze=True)


    #all_runs = np.concatenate([runs_cov, runs_xcov])
    #all_runs = runs_xcov

    #for run in runs_cov:
    #    dump_covs(exp_dir(run))

    #gen_runexp(all_runs, "python python/bcm/treegp/bcm/bcmopt.py", "runexp3.sh", analyze=False)
    #gen_runexp(all_runs, "python python/bcm/treegp/bcm/bcmopt.py", "analyze3.sh", analyze=True)

    # get fixedsize runs
    # get variable runs
    # conglomerate them into a list of run params
    # call gen_runexp twice to generate a run script and an analysis script

    #plot_models_fixedsize(lscale=0.4, obs_std=0.1)
    #plot_models_growing()
    #vis_points(d="bcmopt_experiments/5000_5500_000250_0.05_0.02_0.000_50_l-bfgs-b_x_-1/", y_target=1, sdata_file="bcmopt_experiments/synthetic_datasets/5500_5000_0.05_0.020_50_1001.pkl")

    #plot_models_fixedsize(lscale=0.1, obs_std=0.02)

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
