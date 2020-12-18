from pathlib import Path
import sys

def config_datasphere(force=False):
    homedir = Path.home()
    DATASPHERE = str(homedir) == "/root"

    if DATASPHERE or force:  # if Datasphere
        print("DATASPHERE")
        sys.path.append(str(Path.cwd()/'riiidNew'))
        
        global preprocess_train_data, bayesian_catboost_searchCV
        from preprocess import preprocess_train_data
        from catboost_bayesian_search import bayesian_catboost_searchCV
        
        global dtypes, target 
        from config import dtypes, target

        global features, cat_features
        from config import features, cat_features

        global prior_params, pds, pds_dtypes
        from config import prior_params, pds, pds_dtypes
        


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

features = (
    'content_id', 'prior_question_elapsed_time',
    'prior_question_had_explanation', 'user_correctness',
    'part', 'content_count'
)

cat_features = (
    'prior_question_had_explanation',
    'part',
)

# Catboost initial parameters, to overload next by pds for bayesian search
prior_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'custom_metric': 'AUC:hints=skip_train~false',

    'task_type': 'GPU' if str(homedir) == "/root" else 'CPU',
    # 'task_type': 'GPU' if torch.cuda.is_available() else 'CPU',
    'grow_policy': 'Lossguide',
    'iterations': 2,
    'learning_rate': 4e-3,
    'random_seed': 0,
    'l2_leaf_reg': 5e-1,
    'depth': 10,
    'max_leaves': 10,
    'border_count': 128,
    'verbose': 150,
    'od_type': 'Iter',
    'od_wait': 50,
}

# catboost params intervals for bayesian search
pds = {
    'iterations': [5, 5],
    'l2_leaf_reg': [2e-1, 5e2],
    'depth': [8, 25],
    'max_leaves': [10, 150],
    'border_count': [50, 300],

}

# Parameters dtypes to adjust with catboost
pds_dtypes = {
    'iterations': int,
    'l2_leaf_reg': float,
    'depth': int,
    'max_leaves': int,
    'border_count': int,

}