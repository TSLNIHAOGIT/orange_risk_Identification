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

path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/tianchi_competition/orange_risk_Identification/'
test=pd.read_csv(path+'test.csv')
train=pd.read_csv(path+'train.csv')
train_labels = train['Tag']
train = train.drop(['UID','Tag'],axis=1)

X_loc_test = test.drop('UID',axis=1)




# def identify_zero_importance_(features, labels, eval_metric, task='classification',
#                                   n_iterations=10, early_stopping=True):
#     # One hot encoding
#     # features = pd.get_dummies(features)
#
#     # Extract feature names
#     feature_names = list(features.columns)
#
#     # Convert to np array
#     features = np.array(features)
#     labels = np.array(labels).reshape((-1,))
#
#     # Empty array for feature importances
#     feature_importance_values = np.zeros(len(feature_names))
#
#     print('Training Gradient Boosting Model\n')
#
#     # Iterate through each fold
#     for _ in range(n_iterations):
#
#         if task == 'classification':
#             model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=2)
#
#         elif task == 'regression':
#             model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbose=2)
#
#         else:
#             raise ValueError('Task must be either "classification" or "regression"')
#
#         # If training using early stopping need a validation set
#         if early_stopping:
#
#             train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels,
#                                                                                           test_size=0.15)
#
#             # Train the model with early stopping
#             model.fit(train_features, train_labels, eval_metric=eval_metric,
#                       eval_set=[(valid_features, valid_labels)],
#                       early_stopping_rounds=100, verbose=-1)
#
#             # Clean up memory
#             gc.enable()
#             del train_features, train_labels, valid_features, valid_labels
#             gc.collect()
#
#         else:
#             model.fit(features, labels)
#
#         # Record the feature importances
#         feature_importance_values += model.feature_importances_ / n_iterations
#
#     print('train finished')
#     feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
#
#     # Sort features according to importance
#     feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)
#
#     # Normalize the feature importances to add up to one
#     feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances[
#         'importance'].sum()
#     feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
#
#     # Extract the features with zero importance
#     record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
#
#     to_drop = list(record_zero_importance['feature'])
#
#     feature_importances = feature_importances
#     record_zero_importance = record_zero_importance
#     # removal_ops['zero_importance'] = to_drop
#
#     print('\n%d features with zero importance.\n' % len(to_drop))
#
# identify_zero_importance_(train, train_labels, eval_metric='auc', task='classification',
#                                  n_iterations=10, early_stopping = True)




from feature_selector import FeatureSelector
# Features are in train and labels are in train_labels
fs = FeatureSelector(data = train, labels = train_labels)

# #缺失值统计
# fs.identify_missing(0.5)
# df_miss_value=fs.missing_stats.sort_values('missing_fraction',ascending=False)
# print('df_miss_value',df_miss_value.head(15))
# missing_features = fs.ops['missing']
# print('missing_features to remove',missing_features[:20])
#
# #单值特征统计
# fs.identify_single_unique()
# print('fs.plot_unique()',fs.plot_unique())
#
#
# fs.identify_collinear(0.95)
# print('plot_collinear()',fs.plot_collinear())
#
# # list of collinear features to remove
# collinear_features = fs.ops['collinear']
# print('collinear_features',collinear_features)
#
# # dataframe of collinear features
# df_collinear_features=fs.record_collinear.sort_values('corr_value',ascending=False)
# print('df_collinear_features',df_collinear_features.head())


# #零重要度特征统计
# # Pass in the appropriate parameters
# fs.identify_zero_importance(task = 'classification',
#  eval_metric = tpr_weight_funtion_lc,
#  n_iterations = 10,
#  early_stopping = True)
# # list of zero importance features
# zero_importance_features = fs.ops['zero_importance']
# print('zero_importance_features',zero_importance_features)



# #低重要度特征统计
# fs.identify_low_importance(cumulative_importance = 0.99)
# df_low_importance=fs.feature_importances
# print(df_low_importance.sort_values('importance',ascending=False).head(20))




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

