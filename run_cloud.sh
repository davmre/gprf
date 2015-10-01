#/bin/bash

source /home/sigvisa/.bash_profile
source /home/sigvisa/.virtualenvs/sigvisa/bin/activate

cd /home/sigvisa/python/gprf/
python gprfopt.py $@ >> /home/sigvisa/gprf_log.txt 2>&1
