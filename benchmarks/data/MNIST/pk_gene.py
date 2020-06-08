import sys
sys.path.append('../../../')
import pickle
from data import get_multitask_experiment

pkl = get_multitask_experiment('MNIST',5)
with open('mnist.pk', 'wb') as f:
    pickle.dump(pkl, f)