import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/LightGBM/python-package')

import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

import gc

import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import gc

from sklearn.exceptions import NotFittedError

from itertools import chain


from sklearn.exceptions import NotFittedError

from itertools import chain
from evaluation import tpr_weight_funtion_lc

path0='../results/'
test=pd.read_csv(path0+'test.csv')
train=pd.read_csv(path0+'train.csv')

print('tag_value_counts',train['Tag'].value_counts())

train_labels = train['Tag']
train = train.drop(['UID','Tag'],axis=1)

X_loc_test = test.drop('UID',axis=1)








from feature_selector import FeatureSelector
# Features are in train and labels are in train_labels
fs = FeatureSelector(data = train, labels = train_labels)

#缺失值统计
fs.identify_missing(0.5)
df_miss_value=fs.missing_stats.sort_values('missing_fraction',ascending=False)
print('df_miss_value',df_miss_value.head(15))
missing_features = fs.ops['missing']
print('missing_features to remove',missing_features[:20])

#单值特征统计
fs.identify_single_unique()
print('fs.plot_unique()',fs.plot_unique())


fs.identify_collinear(0.95)
print('plot_collinear()',fs.plot_collinear())

# list of collinear features to remove
collinear_features = fs.ops['collinear']
print('collinear_features',collinear_features)

# dataframe of collinear features
df_collinear_features=fs.record_collinear.sort_values('corr_value',ascending=False)
print('df_collinear_features',df_collinear_features.head(50))


#零重要度特征统计
# Pass in the appropriate parameters
fs.identify_zero_importance(task = 'classification',
 eval_metric = tpr_weight_funtion_lc,
 n_iterations = 10,
 early_stopping = True)
# list of zero importance features
zero_importance_features = fs.ops['zero_importance']
print('zero_importance_features',zero_importance_features)



#低重要度特征统计
fs.identify_low_importance(cumulative_importance = 0.99)
df_low_importance=fs.feature_importances
print(df_low_importance.sort_values('importance',ascending=False).head(20))




#一次行运行所有函数
print('go')
fs.identify_all(selection_params = {'missing_threshold': 0.7,
 'correlation_threshold': 0.99,
 'task': 'classification',
 'eval_metric': tpr_weight_funtion_lc,
 'cumulative_importance': 0.999})


#移除特征
# Remove the features from all methods (returns a df)
left_feature,removed_feature = fs.remove(methods = ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance'],keep_one_hot =True)
print('left_feature\n ',left_feature.columns,left_feature.shape )
print('emoved_feature\n',len(removed_feature),'\n',removed_feature,)

