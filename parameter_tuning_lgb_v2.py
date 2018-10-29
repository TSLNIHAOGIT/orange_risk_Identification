import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from evaluation import metric_self,metric_scores_self

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from random import shuffle
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold
from evaluation import tpr_weight_funtion_lgb_cv,tpr_weight_funtion_lc

MAX_EVALS = 1000
N_FOLDS = 5

path0='./results/'


test=pd.read_csv(path0+'test_select.csv')
train=pd.read_csv(path0+'train_select.csv')
labels=train['Tag']

scale_pos_weight_compute=len(labels[labels==0])/len(labels[labels==1])


res = test.loc[:, ['UID']]
y_loc_train = train['Tag'].values
X_loc_train = train.drop(['UID','Tag'],axis=1).values
X_loc_test = test.drop('UID',axis=1).values


import pandas as pd
import numpy as np

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold
import random
from timeit import default_timer as timer






path='./'

train = X_loc_train
# test = data[data['ORIGIN'] == 'test']

# Extract the labels and format properly
train_labels = y_loc_train
# test_labels = np.array(test['CARAVAN'].astype(np.int32)).reshape((-1,))


# Convert to numpy array for splitting in cross validation
features = train
test_features = X_loc_test
labels = train_labels



################################################################################
# Create a lgb dataset
train_set = lgb.Dataset(features, label = labels)

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
# out_file = path+'params_results/gbm_trials.csv'

def objective(params, n_folds=N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)

    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])

    start = timer()


    print('params',params)
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round=10000, nfold=n_folds,
                        stratified=True,

                        early_stopping_rounds=100,
                        feval=tpr_weight_funtion_lgb_cv,
                        seed=50,
                        verbose_eval=1#用于一次显示第几个几个cv-agg，cv_agg's TPR: -0.777223 + 0.0223719，均值和标准差
                        )

    # print('cv_results',type(cv_results),'\n',cv_results)
    print('df_cv_results\n',pd.DataFrame(cv_results))

    run_time = timer() - start

    # Extract the best score
    best_score = np.min(cv_results['TPR-mean'])

    # Loss must be minimized
    loss = best_score

    TPR_stdv=cv_results['TPR-stdv'][cv_results['TPR-mean'].index(best_score)]
    print('TPR_stdv',TPR_stdv)


    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmin(cv_results['TPR-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss,TPR_stdv, params, ITERATION, n_estimators, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'TPR_stdv':TPR_stdv,'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'train_time': run_time, 'status': STATUS_OK}




from hyperopt import hp
from hyperopt.pyll.stochastic import sample



##########################################################################
# Create the learning rate
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}

# Discrete uniform distribution
num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}


# boosting type domain
boosting_type = {'boosting_type': hp.choice('boosting_type',
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}])}

# Draw a sample
params = sample(boosting_type)

# Retrieve the subsample if present otherwise set to 1.0
subsample = params['boosting_type'].get('subsample', 1.0)

# Extract the boosting type
params['boosting_type'] = params['boosting_type']['boosting_type']
params['subsample'] = subsample


##########################################################################

#to  Complete Bayesian Domain,Now we can define the entire domain

# Define the search space
space = {
    'scale_pos_weight':hp.uniform('scale_pos_weight', 1,2*scale_pos_weight_compute),#不平衡数据集需要设置负样本与正样本之比，；平衡数据集则不用设置
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0)
}



# # Sample from the full space
# '''Let's sample from the domain (using the conditional logic)
# to see the result of each draw. Every time we run this code,
# the params_results will change.'''
# x = sample(space)
#
# # Conditional logic to assign top-level keys
# subsample = x['boosting_type'].get('subsample', 1.0)
# x['boosting_type'] = x['boosting_type']['boosting_type']
# x['subsample'] = subsample
# x
#
# x = sample(space)
# subsample = x['boosting_type'].get('subsample', 1.0)
# x['boosting_type'] = x['boosting_type']['boosting_type']
# x['subsample'] = subsample
# x


# Optimization Algorithm

from hyperopt import tpe

# optimization algorithm
tpe_algorithm = tpe.suggest


# Results History
from hyperopt import Trials

# Keep track of params_results
bayes_trials = Trials()

# File to save first params_results
# very time the objective function is called,
# it will write one line to this file.
# Running the cell above does clear the file though.
out_file = path+'params_results/gbm_trials_temp.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'TPR_stdv','params', 'iteration', 'estimators', 'train_time'])
of_connection.close()

# Bayesian Optimization
from hyperopt import fmin


# %%capture

# Global variable
global  ITERATION

ITERATION = 0

# Run optimization
#set space  as params to send to objective

algo = partial(tpe.suggest,n_startup_jobs=-1)
best = fmin(fn = objective, space = space, algo = algo,   # algo= tpe.suggest,
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))


print('best',best)

# Sort the trials with lowest loss (highest AUC) first
bayes_trials_results = sorted(bayes_trials.params_results, key = lambda x: x['loss'])
print('bayes_trials_results[:2]',bayes_trials_results[:2])

#################################################################

results = pd.read_csv(path0+path+'params_results/gbm_trials.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
print('params_results.head()',results.head())




import ast

# Convert from a string to a dictionary
ast.literal_eval(results.loc[0, 'params'])


 # Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, n_jobs = -1,
                                       objective = 'binary', random_state = 50, **best_bayes_params)
best_bayes_model.fit(features, labels)

# Evaluate on the testing data
preds = best_bayes_model.predict_proba(test_features)[:, 1]
res['Tag']=preds
import datetime
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
res[['UID', 'Tag']].to_csv(path0+"lgb_all_data_best_para_%s.csv" % now, index=False)

# print('The best model from Bayes optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(test_labels, preds)))
print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))




















