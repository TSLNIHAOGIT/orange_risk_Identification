from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import datetime
import xgboost as xgb
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import warnings
import time
import pandas as pd
import numpy as np
import os
import re
import pandas as pd
import urllib.parse
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import gc
###################################
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,BaggingClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import \
    make_blobs  # 聚类数据生成器其参数设置详见：https://blog.csdn.net/kevinelstri/article/details/52622960
from sklearn.metrics import log_loss,roc_auc_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.svm import SVC
from evaluation import tpr_weight_funtion_self,tpr_weight_funtion_lc,tpr_weight_funtion2,\
    metric_self,log_loss_temp,evalmcc_min



import warnings

warnings.filterwarnings('ignore')


test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')
#填充缺失值
test=test.fillna(-1)
train=train.fillna(-1)

print(train.info())


res = test.loc[:, ['UID']]

y_loc_train = train['Tag'].values
X_loc_train = train.drop(['UID','Tag'],axis=1).values

X_loc_test = test.drop('UID',axis=1).values




models=[
        ######## First level ########
        [
        ExtraTreesClassifier(n_estimators=2000, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=2000, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.9, max_depth=6, n_estimators=2000
                                             ,
                                             criterion='friedman_mse',
                                             init=None,
                                             loss='deviance',
                                             max_features=None, max_leaf_nodes=None,
                                             min_impurity_decrease=1e-07, min_samples_leaf=1,
                                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                                              presort='auto', random_state=4,
                                              verbose=0, warm_start=False
                                             ),

        xgb.XGBClassifier(
                            # eval_metric="auc",
                            max_depth=20, learning_rate=0.05,n_estimators=2000, silent=True,
                            objective="binary:logistic", booster='gbtree',n_jobs=10,  gamma=0, min_child_weight=5,
                            max_delta_step=0, subsample=0.8, colsample_bytree=1, colsample_bylevel=1,
                            reg_alpha=3, reg_lambda=5, scale_pos_weight=1,random_state=1000,),


         # RandomForestClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
         # ExtraTreesClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
         # GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
         # LogisticRegression(random_state=1),

         ],
        ######## Second level ########
        [
            lgb.LGBMClassifier(
                                      # metric='auc',
                                      boosting_type='gbdt', num_leaves=48, max_depth=-1,
                                      learning_rate=0.05, n_estimators=2000,
                                      max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                                      min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=1, reg_alpha=3, reg_lambda=5, random_state=1000, n_jobs=10,
                                      silent=True, ),
            BaggingClassifier(verbose=1,n_jobs=-1, n_estimators=2000, max_samples=1.0,
                                    max_features=1.0, bootstrap=True, bootstrap_features=False, random_state=1),
        ],
            # RandomForestClassifier (n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)],


        ######## Third level ########
        [
            RandomForestClassifier (n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
         # LogisticRegression(random_state=1),
         ]
       ]
from pystacknet.pystacknet import StackNetClassifier

model=StackNetClassifier(models, metric=metric_self, folds=5,
	restacking=False,use_retraining=True, use_proba=True,
	random_state=12345,n_jobs=-1, verbose=1)

model.fit(X_loc_train,y_loc_train)
preds=model.predict_proba(X_loc_test)[:, 1]

predict_result = test.loc[:, ['UID']]
predict_result['Tag'] =preds
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['UID', 'Tag']].to_csv("lgb_stacknet%s.csv" % now, index=False)
print(predict_result.head())