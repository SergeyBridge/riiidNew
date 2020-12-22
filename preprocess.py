#%%

import pandas as pd
import catboost
from catboost import CatBoostClassifier, cv
# from torch import cuda

import numpy as np
import scipy
from bayes_opt import BayesianOptimization



def preprocess_train_data(df, questions_df, target, dtypes):
    # Exclude lectures
    df = df[df[target] != -1].reset_index(drop=True, inplace=False)
    # Fill NaN values in the 'prior_question_had_explanation' columns
    df['prior_question_had_explanation'].fillna(False, inplace=True)
    # Set type
    df = df.astype(dtypes)

    # Answer for the previous questions of users
    df['lag'] = df.groupby('user_id')[target].shift()
    # For each user (groupby('user_id')), compute the cummulative number of correct answers and number answers in general
    groupby = df.groupby('user_id')['lag']
    cum = groupby.agg(['cumsum', 'cumcount'])

    # User correctness (measure the users' learning progress)
    df['user_correctness'] = cum['cumsum'] / cum['cumcount']
    # Drop the 'lag' feature
    df.drop(columns=['lag'], inplace=True)
    df.head()

    # Overall correctness of users
    user_agg = df.groupby('user_id')[target].agg(['sum', 'count'])
    # Overall difficulty of questions
    content_agg = df.groupby('content_id')[target].agg(['sum', 'count'])

    # Take only 24 last observations of each user
    df = df.groupby('user_id').tail(24).reset_index(drop=True)

    df = pd.merge(df, questions_df, left_on='content_id', right_on='question_id', how='left')
    df.drop(columns=['question_id'], inplace=True)

    # How many questions have been answered in each content ID?
    df['content_count'] = df['content_id'].map(content_agg['count']).astype('int32')
    # How hard are questions in each content ID?
    df['content_id'] = df['content_id'].map(content_agg['sum'] / content_agg['count'])

    return user_agg, content_agg, df



def search_cv(train_set, prior_params, pds, pds_dtypes,init_points=3, n_iter=7, verbose=0):

    def catboost_hyperparams(**dict_):
        params = prior_params.copy()
        dict_ = {key: pds_dtypes[key](val) for key, val in dict_.items()}
        params.update(dict_)

        # Fitting
        scores = cv(train_set, params,
                    plot=False,
                    # metric_period=10,
                    type='TimeSeries',
                    fold_count=3)

        return np.max(scores["test-AUC-mean"])

    optimizer = BayesianOptimization(catboost_hyperparams, pds, random_state=0, verbose=verbose)
    optimizer.maximize(init_points, n_iter)
    return optimizer.max
