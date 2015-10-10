#/bin/bash

source /home/sigvisa/.bash_profile
source /home/sigvisa/.virtualenvs/sigvisa/bin/activate

cd /home/sigvisa/python/gprf/
ulimit -Sv 17179869184 #set 16gb ram limit so we don't overload other processes
python $@ >> /home/sigvisa/gprf_log.txt 2>&1
