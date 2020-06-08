import sys
sys.path.append('../../../')
import pickle
from data import get_multitask_experiment

pkl = get_multitask_experiment('CIFAR',5)
with open('cifar.pk', 'wb') as f:
    pickle.dump(pkl, f)