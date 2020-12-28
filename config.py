from pathlib import Path
import numpy as np
import sys
from catboost.utils import get_gpu_device_count

dtypes = {
    'row_id': 'int64',
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'bool',
    'user_answer': 'int8',
}

target = 'answered_correctly'

homedir = Path.home()

new_features = {
    'question_difficulty': np.float,
    'user_correctness': np.float32,
    'content_count': np.int32

}

features = {
    'content_id': np.int32,
    'prior_question_elapsed_time': np.int32,
    'prior_question_had_explanation': np.bool,
    'part': np.int32,
}
features.update(new_features)

cat_features = {
    'prior_question_had_explanation': np.bool,
    'part': np.int,
}



# Catboost initial parameters, to overload next by pds for bayesian search
prior_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'custom_metric': 'AUC:hints=skip_train~false',

    #'task_type': 'GPU' if str(homedir) == "/root" else 'CPU',
    'task_type': 'GPU' if str(Path.home()) == "/home/riiid" else 'CPU',  # if virtual machine then GPU
    # 'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
    'grow_policy': 'Lossguide',
    'iterations': 2,
    'thread_count': -1,
    'learning_rate': 0.1,  # 3e-2,
    'random_seed': 0,
    'bootstrap_type': 'Bayesian',
    'l2_leaf_reg': 5e-1,
    'depth': 10,
    'max_leaves': 10,
    'border_count': 128,
    'verbose': 250,
    'od_type': 'Iter',
    'od_wait': 50,
}

# catboost params intervals for bayesian search
pds = {
    'l2_leaf_reg': [2e-2, 5e2],
    'bagging_temperature': [0.3, 5],
    'depth': [8, 25],
    'max_leaves': [8, 150],
    'border_count': [50, 300],

}

# Parameters dtypes to adjust with catboost
pds_dtypes = {
    'iterations': int,
    'l2_leaf_reg': float,
    'bagging_temperature': float,
    'depth': int,
    'max_leaves': int,
    'border_count': int,

}