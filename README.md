# Gaussian Process Random Fields

This is code for the NIPS 2015 paper by David Moore and Stuart Russell. Aside from the usual dependencies (numpy, scipy, matplotlib), it depends on:
- the [treegp](https://github.com/davmre/treegp) package, which contains C++ implementations of several distance functions, kernel functions and their derivatives. In particular, it implements the great-circle distance used in the seismic experiments. In case compatibility is broken at some future point, commit `a0aa7ae65a4b9144a499016bbf0ccaf0c611cc0d` is known to work with this code. 
- [GPy](https://github.com/SheffieldML/GPy), for comparisons to sparse GP-LVM. Experiments were run using version 0.6.0.

Individual synthetic experiments from the paper can be reproduced by running `gprfopt.py` with appropriate options. For example,

    python gprfopt.py --n=10000 --seed=0 --yd=50 --lscale=0.06 --obs_std=0.02 --noise_var=0.01 --method=l-bfgs-b  --local_dist=1.0 --nblocks=100 --task=x --maxsec=18000

will sample a synthetic problem with 10000 points, random seed 0, 50-dimensional output, SE kernel lengthscale 0.06 (note a small difference from the paper: this implementation scales the world to always lie within the unit square, so larger problems correspond to smaller lengthscales) and positional noise stddev 0.02, and output noise variance 0.01, and then attempt to solve this problem by running L-BFGS in a local GP model (local_dist=1.0 specifies a purely local GP, local_dist < 1.0 defines a GPRF and the specific value does not matter) with 100 blocks, solving only for the X locations (not kernel params), and running for a maximum time of 18000 seconds. The results will be saved under the home directory in `~/gprf_experiments/`. A subdirectory is created for each experiment, and the file `results.txt` contains the objective value and mean location error at each step (along with other quantities). 

After running a synthetic experiment, you can visualize the results, e.g., for the previous example,

    python gprfopt_analyze.py vis ~/gprf_experiments/10000_10500_100_0.060000_0.020000_1.0000_50_l-bfgs-b_x_-1_0.0100_s0_gprf0/ ~/gprf_experiments/synthetic_datasets/10500_10000_0.060000_0.020000_50_0.pkl 0

will generate a series of images, one for each optimization step, and attempt to stich them into a video. 

The seismic dataset, consisting of an array of (lon, lat, depth) values, is stored in `sorted_isc.npy`. Individual seismic experiments can be reproduced by `run_seismic.py`. For example,

    python run_seismic.py --obs_std=20.0 --rpc_blocksize=210  --task=xcov --threshold=0.6  --maxsec=345600

will generate a set of noisy location observations with stddev 20km,
partitioned into blocks of at most 210 points (allowing some leeway
since the principle axis tree recursively splits the dataset and may
not obtain the precise block size requested), and run inference to
recover both X locations and kernel hyperparameters, using a GPRF
containing an edge between any two blocks for which some initial cross-covariance 
value is at least 0.6 (corresponding to one kernel lengthscale), and running
for at most 345600 seconds (four days). The results will be saved under the home 
directory in `~/seismic_experiments/`. 

To automatically generate the full set of synthetic experiments from the paper, run `gprfopt_analyze.py` with no arguments: this will generate Bash scripts `run_truegp.sh`, `run_fitc.sh`, and `run_eighty.sh`. The iPython notebook `gprf_camera_plot.ipynb` was used to generate the plots in the paper, based on results which are included in the tarballs `gprf_experiments.tgz` and `seismic_experiments.tgz`. 

For questions about the paper or if you have difficulty running the code (this is likely!) or reproducing the experimental results, please contact Dave Moore at dmoore@cs.berkeley.edu. 
