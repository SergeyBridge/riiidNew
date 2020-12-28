#%%

import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier, cv
# from torch import cuda

import numpy as np
import scipy
from bayes_opt import BayesianOptimization

concat_dtypes = {
    'row_id': np.int64,
    'timestamp': np.int64,
    'user_id':  np.int32,
    'content_id': np.int16,
    'content_type_id': np.int8,
    'task_container_id': np.int16,
    'answered_correctly': np.int8,
    'prior_question_elapsed_time': np.float32,
    'prior_question_had_explanation': np.bool_,
    'user_answer': np.int8,
    'user_correctness': np.float64,
    'question_id': np.int64,
    'part': np.int64,
}

def preprocess_train_data(df, questions_df, target, dtypes):
    # Exclude lectures
    df = df[df['content_type_id'] == 0].reset_index(drop=True, inplace=False)

    # Fill NaN values in the 'prior_question_had_explanation' columns
    df['prior_question_had_explanation'].fillna(False, inplace=True)
    # Set type
    df = df.astype(dtypes)

    # Answer for the previous questions of users
    df['lag'] = df.groupby('user_id')[target].shift()

    # compute the cummulative number of correct answers and number answers in general
    groupby = df.groupby('user_id')['lag']
    cum = groupby.agg(['cumsum', 'cumcount'])

    # User correctness (measure the users' learning progress)
    df['user_correctness'] = cum['cumsum'] / cum['cumcount']
    # Drop the 'lag' feature
    df.drop(columns=['lag'], inplace=True)

    # Overall correctness of users
    user_agg = df.groupby('user_id')[target].agg(['sum', 'count'])
    # Overall difficulty of questions
    content_agg = df.groupby('content_id')[target].agg(['sum', 'count'])

    # Take only 24 last observations of each user
    # df = df.groupby('user_id').tail(24).reset_index(drop=True)
    df = pd.concat(
        [df.reset_index(drop=True),
         questions_df.reindex(df['content_id'].values).reset_index(drop=True)],
         axis=1, sort=False, join="inner")

    # df_merge = pd.merge(df, questions_df, left_on='content_id', right_on='question_id', how='left')
    df.drop(columns=['question_id'], inplace=True)

    # How many questions have been answered in each content ID?
    df['content_count'] = df['content_id'].map(content_agg['count']).astype('int32')
    # How hard are questions in each content ID?
    # df['content_id'] = df['content_id'].map(content_agg['sum'] / content_agg['count'])
    df.prior_question_elapsed_time = df.prior_question_elapsed_time // 2000
    df['question_difficulty'] = df.content_id.map(content_agg['sum'] / content_agg['count'])

    return user_agg, content_agg, df


def divide_nan(numerator, denominator):
    return float(np.divide(numerator, denominator,
                 out=np.array(np.nan), where=denominator!=0))
