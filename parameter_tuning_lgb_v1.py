import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss
from evaluation import metric_self,metric_scores_self
# path="/Users/ozintel/Downloads/Tsl_python_progect/local_ml/LightGBM/examples/regression/"
# print("load data")
# df_train=pd.read_csv(path0+path+"regression.train",header=None,sep='\t')
# df_test=pd.read_csv(path0+path+"regression.train",header=None,sep='\t')
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

path0='./results/'



test=pd.read_csv(path0+'test.csv')
train=pd.read_csv(path0+'train.csv')
y_loc_train = train['Tag'].values
X_loc_train = train.drop(['UID','Tag'],axis=1)
# X_loc_test = test.drop('UID',axis=1)

def GBM(argsDict):
    # print('argsDict',argsDict)
    max_depth = int(argsDict["max_depth"] )
    min_child_weight = int(argsDict["min_child_weight"])
    gamma=argsDict['gamma']
    reg_lambda=argsDict['reg_lambda']
    colsample_bytree=argsDict['colsample_bytree']
    subsample = argsDict["subsample"]

    num_parallel_tree=int(argsDict['num_parallel_tree'])
    # n_estimators = int(argsDict['n_estimators'] )
    learning_rate = argsDict["learning_rate"]



    gbm = xgb.XGBClassifier(n_jobs=-1,    #进程数
                            max_depth=max_depth,  #最大深度
                            n_estimators=1000,   #树的数量
                            learning_rate=learning_rate, #学习率
                            subsample=subsample,      #采样数
                            min_child_weight=min_child_weight,   #孩子数
                            # max_delta_step = 10,  #10步不降则停止
                            objective="binary:logistic",
                            gamma=gamma,
                            reg_lambda=reg_lambda,
                            colsample_bytree=colsample_bytree,
                            num_parallel_tree=num_parallel_tree,
                            verbose=2,
                            silent=True

                            )

    for key in argsDict:
        print(key,gbm.get_params()[key])
    print('\n\n')

    metric = cross_val_score(gbm,X_loc_train,y_loc_train,cv=5,scoring=metric_scores_self,n_jobs=-1).mean()
    print('metric',metric)
    return metric

#hp.randint(label, upper) 返回从[0, upper)的随机整数
space = {
       "max_depth":hp.quniform("max_depth",1,10,1),
         "min_child_weight":hp.quniform("min_child_weight",1,5,1), #
        'gamma':hp.uniform('gamma', 0, 1),
         'reg_lambda':hp.uniform('reg_lambda', 0, 1),


        'subsample':hp.uniform('subsample', 0.5, 1),
        'colsample_bytree':hp.uniform('colsample_bytree', 0.6, 1),
        'num_parallel_tree':hp.quniform('num_parallel_tree',1,4,1),
         #"n_estimators":hp.randint("n_estimators",10),  #[0,1,2,3,4,5] -> [50,]
         "learning_rate":hp.uniform("learning_rate",0,1),  #[0,1,2,3,4,5] -> 0.05,0.06


        }
algo = partial(tpe.suggest,n_startup_jobs=-1)
#输入是gbm就包含mertric评价
best = fmin(GBM,space,algo=algo,max_evals=500)#max_evals表示想要训练的最大模型数量，越大越容易找到最优解
print('*****************************')
print('best\n',best)#best调用时，任然返回；字典+两个换行+metric
print ('GBM(best)',GBM(best))


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

