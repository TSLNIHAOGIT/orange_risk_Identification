import json
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from evaluation import metric_self,metric_scores_self,tpr_weight_funtion_xgb_cv
# path="/Users/ozintel/Downloads/Tsl_python_progect/local_ml/LightGBM/examples/regression/"
# print("load data")
# df_train=pd.read_csv(path+"regression.train",header=None,sep='\t')
# df_test=pd.read_csv(path+"regression.train",header=None,sep='\t')
#
#
# print(df_train.shape,df_test.shape)
# # print('df_train.dtypes',df_train.dtypes)
# # print('df_train.head()',df_train.head())
# print('df_train.columns',df_train.columns)
# y_train = df_train[0].values
# print('type(y_train)',type(y_train))
# y_test = df_test[0].values
# X_train = df_train.drop(0, axis=1).values
# X_test = df_test.drop(0, axis=1).values
#
#
# from hyperopt import hp,fmin, rand, tpe, space_eval
#
# print("LGB test")
# clf = lgb.LGBMClassifier(
#     num_class=4,
#     max_depth=-1,
#     n_estimators=5000,
#     objective='multiclass',
#     learning_rate=0.01,
#     num_leaves=65,
#     n_jobs=-1,
# )

from hyperopt import  hp,fmin, rand, tpe, space_eval

# def q (args) :
#     x, y = args
#     return x**2-2*x+1 + y**2
#
# space = [hp.randint('x', 5), hp.randint('y', 5)]
# best = fmin(q,space,algo=rand.suggest,max_evals=10)
# print(best)


#coding:utf-8
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
import csv

MAX_EVALS = 1000
N_FOLDS = 5


path='./'

path0='./results/'



test=pd.read_csv(path0+'test_select.csv')
train=pd.read_csv(path0+'train_select.csv')


res = test.loc[:, ['UID']]
y_loc_train = train['Tag'].values
X_loc_train = train.drop(['UID','Tag'],axis=1)
X_loc_test = test.drop('UID',axis=1)



# Keep track of params_results
bayes_trials = Trials()

# File to save first params_results
# very time the objective function is called,
# it will write one line to this file.
# Running the cell above does clear the file though.

out_file = path+'params_results/xgb_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
# Write the headers to the file
writer.writerow(['loss', 'TPR_stdv','params', 'iteration', 'estimators', 'train_time'])
of_connection.close()






train = X_loc_train
# test = data[data['ORIGIN'] == 'test']

# Extract the labels and format properly
train_labels = y_loc_train
# test_labels = np.array(test['CARAVAN'].astype(np.int32)).reshape((-1,))


# Convert to numpy array for splitting in cross validation
features = train
test_features = X_loc_test
labels = train_labels

# Create a lgb dataset
train_set = xgb.DMatrix(features, label = labels)

import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer




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
    for parameter_name in ['max_depth', 'subsample_for_bin', 'min_child_samples','min_child_weight','num_parallel_tree']:
        params[parameter_name] = int(params[parameter_name])

    start = timer()

    print('params',params)
    # Perform n_folds cross validation
    cv_results = xgb.cv(params, train_set,
                        num_boost_round=3000,
                        nfold=n_folds,
                        stratified=True,
                        early_stopping_rounds=100,
                        feval=tpr_weight_funtion_xgb_cv,
                        seed=50,
                        verbose_eval=True,

                        )

    print('cv_results\n',type(cv_results),'\n',cv_results)

    run_time = timer() - start

    # Extract the best score
    best_score = np.min(cv_results['test-TPR-mean'])

    # Loss must be minimized
    loss = best_score

    TPR_std = cv_results[cv_results['test-TPR-mean']==best_score]['test-TPR-std'].values[0]
    print('TPR_stdv', TPR_std)


    # Boosting rounds that returned the highest cv score
    n_estimators = int(np.argmin(cv_results['test-TPR-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss,TPR_std, params, ITERATION, n_estimators, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss,'TPR_std':TPR_std, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators,
            'train_time': run_time, 'status': STATUS_OK}


#hp.randint(label, upper) 返回从[0, upper)的随机整数
space = {
      'scale_pos_weight':hp.uniform('scale_pos_weight', 1,10),
       "max_depth":hp.quniform("max_depth",1,15,1),
         "min_child_weight":hp.quniform("min_child_weight",1,5,1), #
        'gamma':hp.uniform('gamma', 0, 1),
         'reg_lambda':hp.uniform('reg_lambda', 0, 1),
        'reg_alpha':hp.uniform('reg_alpha', 0, 1),
        'subsample':hp.uniform('subsample', 0.5, 1),
        'colsample_bytree':hp.uniform('colsample_bytree', 0.6, 1),
        'num_parallel_tree':hp.quniform('num_parallel_tree',1,4,1),
         #"n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.uniform("learning_rate",0,1),  #[0,1,2,3,4,5] -> 0.05,0.06


        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'boosting_type': hp.choice('boosting_type',
                               [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                {'boosting_type': 'goss', 'subsample': 1.0}]),
        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
        'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),

}




# Global variable
global  ITERATION

ITERATION = 0

algo = partial(tpe.suggest,n_startup_jobs=-1)
best = fmin(objective,space,algo=algo,max_evals=300)#max_evals表示想要训练的最大模型数量，越大越容易找到最优解
print('*****************************')
print('best\n',best)



# Sort the trials with lowest loss (highest AUC) first
bayes_trials_results = sorted(bayes_trials.params_results, key = lambda x: x['loss'])
print('bayes_trials_results[:2]',bayes_trials_results[:2])




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
best_bayes_model = xgb.XGBClassifier(n_estimators=best_bayes_estimators, n_jobs = -1,
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




'''
best {'colsample_bytree': 0.9891884967602931, 'gamma': 0.06914881393959088, 'learning_rate': 0.015736469867526304, 'max_depth': 1.0, 'min_child_weight': 2.0, 'num_parallel_tree': 3.0, 'reg_lambda': 0.82733938713753, 'subsample': 0.5038559615304257}
colsample_bytree 0.9891884967602931
gamma 0.06914881393959088
learning_rate 0.015736469867526304
max_depth 1
min_child_weight 2
num_parallel_tree 3
reg_lambda 0.82733938713753
subsample 0.5038559615304257

******************************

metric 0.4177923497267759
GBM(best) 0.4177923497267759
'''

