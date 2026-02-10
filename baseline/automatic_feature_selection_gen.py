import os
import sys

# Get the base directory of the project, which is two levels up from the current file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'VTFS', 'code'))

print(sys.path)
from feature_env import FeatureEvaluator, base_path
from baseline.model.MARLFS import *
from utils.logger import info
import pickle

import warnings

warnings.filterwarnings('ignore')

# baseline_name = {
#     'kbest': gen_kbest,
#     'mrmr': gen_mrmr,
#     'lasso': gen_lasso,
#     'rfe': gen_rfe,
#     'gfs': gen_gfs,
#     'mcdm': gen_mcdm,
#     'sarlfs': gen_sarlfs,
#     'lassonet': gen_lassonet
# }


def gen_auto_feature_selection(fe_, task_name_):
    fe_.train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='raw_train')
    fe_.test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='raw_test')
    max_accuracy, optimal_set, k = gen_marlfs(fe_, N_ACTIONS=2, N_STATES=64, EXPLORE_STEPS=300)
    best_train = fe_.generate_data(optimal_set, 'train')
    best_test = fe_.generate_data(optimal_set, 'test')
    best_train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='marlfs_train')
    best_test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='marlfs_test')
    # for name_, func in baseline_name.items():
    #     p_, optimal_set = func(fe_, k)
    #     best_train = fe_.generate_data(optimal_set,  flag='train')
    #     best_test = fe_.generate_data(optimal_set, flag='test')
    #     best_train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key=f'{name_}_train')
    #     best_test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key=f'{name_}_test')
    return k


def process(task_name_):
    fea_eval = FeatureEvaluator(task_name_)
    gen_auto_feature_selection(fea_eval, task_name_)
    with open(f'{base_path}/history/{task_name_}/fe.pkl', 'wb') as f:
        pickle.dump(fea_eval, f)
        
import argparse
parser = argparse.ArgumentParser(description='PyTorch Experiment')
parser.add_argument('--name', default='openml_586', type=str)
args, _ = parser.parse_known_args()

process(args.name)
