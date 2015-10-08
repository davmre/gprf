#/bin/bash

source /home/sigvisa/.bash_profile
source /home/sigvisa/.virtualenvs/sigvisa/bin/activate

cd /home/sigvisa/python/gprf/
ulimit -Sv 12000000 #set 12gb ram limit so we don't overload other processes
python $@ >> /home/sigvisa/gprf_log.txt 2>&1
