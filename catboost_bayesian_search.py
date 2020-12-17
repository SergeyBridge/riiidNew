import numpy as np
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier, cv


def bayesian_catboost_searchCV(train_set, prior_params, pds, pds_dtypes,
                               init_points=3, n_iter=7, verbose=0):

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


# USE WITH
# print("optimizer_maxCV")
# optimizer_maxCV = bayesian_catboost_searchCV(train_set, prior_params, pds, pds_dtypes,
#                                               init_points=3, n_iter=7, verbose=0)
# optimizer_maxCV

def bayesian_catboost_search(train_set, val_set, prior_params, pds, pds_dtypes,
                             init_points=3, n_iter=7, verbose=0):
    def catboost_hyperparams(**dict_):
        params = prior_params.copy()

        dict_ = {key: pds_dtypes[key](val) for key, val in dict_.items()}


        params.update(dict_)

        # Model definition
        model = CatBoostClassifier(**params)

        # Fitting
        model.fit(train_set, eval_set=val_set, use_best_model=True)

        return np.max(model.get_best_score()["validation"]["AUC"])

    optimizer = BayesianOptimization(catboost_hyperparams, pds, random_state=0, verbose=verbose)
    optimizer.maximize(init_points, n_iter)
    return optimizer.max

# optimzer_max = bayesian_catboost_search(train_set=train_set, val_set=val_set,
#                                         prior_params=prior_params, pds=pds,
#                                         init_points=5, n_iter=7)
# optimzer_max
