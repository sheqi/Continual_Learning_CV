import sys
sys.path.append('../../../')
import pickle
from data import get_multitask_experiment
factors = ['clutter', 'illumination', 'pixel', 'occlusion','sequence']

for factor in factors:
    pkl = get_multitask_experiment('openloris',9,factor=factor)
    with open(factor + '.pk', 'wb') as f:
        pickle.dump(pkl, f)
