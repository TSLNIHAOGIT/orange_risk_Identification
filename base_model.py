import pandas as pd
from sklearn.datasets import load_digits
import sklearn.model_selection
import os
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import pandas as pd
import time
import datetime
import gc
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import re
import pandas as pd
import urllib.parse
from sklearn import preprocessing
from scipy import sparse
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
# from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb
from feature_seletion import IG
from evaluation import tpr_weight_funtion_self
from evaluation import tpr_weight_funtion_lc,evalmcc_min
from xgboost import XGBClassifier
from feature_selector.feature_selector import  FeatureSelector
import ast

info_g=['code2_sum', 'code2', 'ip2_sum', 'ip2_sub_sum', 'ip2', 'ip2_sub', 'code1_sum', 'device1_x', 'code1', 'device2_x', 'device_code2_sum_y', 'mac1_sum_y', 'device_code1_sum_y', 'device1_y', 'device2_y', 'mac1_sum_x', 'device_code2_y', 'mac1_x', 'market_code_sum', 'market_type_sum', 'mac1_y', 'device_code1_y', 'geo_code_y', 'acc_id3_sum', 'acc_id2_sum', 'device_code3_sum_y', 'market_code', 'market_type', 'wifi_sum', 'wifi', 'version', 'device_code3_sum_x', 'device_code3_x', 'acc_id3', 'acc_id2', 'device_code2_sum_x', 'device_code2_x', 'device_code3_y', 'channel', 'geo_code_sum_y', 'device_code1_x', 'geo_code_x', 'mac2_sum', 'mac2', 'device_code1_sum_x', 'os', 'geo_code_sum_x', 'trans_type1', 'trans_type2', 'ip1_sub_y', 'ip1_y', 'amt_src2', 'amt_src1', 'device1_sum_y', 'mode', 'o_t_days', 'ip1_sub_x', 'merchant', 'ip1_x', 'acc_id1', 'acc_id1_sum', 'device2_sum_y', 'ip1_sum_y', 'ip1_sub_sum_y', 'day_y', 'channel_sum', 'amt_src1_sum', 'merchant_sum', 'trans_type1_sum', 'bal', 'amt_src2_sum', 'trans_type2_sum', 'device2_sum_x', 'trans_amt', 'day_x', 'mode_sum', 'os_sum', 'success']


# test=pd.read_csv('test.csv')
# train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')


def get_best_params():
    results = pd.read_csv('results/gbm_trials0.csv')

    # Sort with best scores on top and reset index for slicing
    results.sort_values('loss', ascending=True, inplace=True)
    results.reset_index(inplace=True, drop=True)
    print('results.head()', results.head())



    # Convert from a string to a dictionary
    ast.literal_eval(results.loc[0, 'params'])

    # Extract the ideal number of estimators and hyperparameters
    best_bayes_estimators = int(results.loc[0, 'estimators'])
    best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()

    return best_bayes_estimators,best_bayes_params




# path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/sweet_orange/orange_data'
# label_train=pd.read_csv(os.path.join(path,"tag_TRAIN.csv"),dtype=str)
# test=pd.read_csv('test_cross_feature.csv')
# train=pd.read_csv('train_cross_feature.csv')
# train=pd.merge(label_train,train,how='left',on='UID')
# #填充缺失值

#不填充时效果好，0.987945148966871


print(train.info())


res = test.loc[:, ['UID']]

y_loc_train = train['Tag'].values
X_loc_train = train.drop(['UID','Tag'],axis=1)

X_loc_test = test.drop('UID',axis=1)

##特征选择
print('开始特征选择')
t1=time.time()

fs = FeatureSelector(data = X_loc_train, labels = y_loc_train)
fs.identify_all(selection_params = {'missing_threshold': 0.7,
 'correlation_threshold': 0.99,
 'task': 'classification',
 'eval_metric': tpr_weight_funtion_lc,
 'cumulative_importance': 0.999})

left_feature,removed_feature = fs.remove(methods = ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance'],keep_one_hot =True)

X_loc_train = left_feature.values
X_loc_test = X_loc_test.drop(labels=removed_feature,axis=1).values



#################################################################
# # lgb.LGBMClassifier()
# feature_select=RFE(estimator=RandomForestClassifier(n_estimators=500,
#                  criterion="gini",
#                  max_depth=8,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_split=1e-7,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=-1,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False,
#                  class_weight=None), n_features_to_select=int(X_loc_train.shape[1] * 0.80))
# feature_select.fit(X_loc_train, y_loc_train)
# X_loc_train = feature_select.transform(X_loc_train)
# X_loc_test = feature_select.transform(X_loc_test)


# ig=IG(X_loc_train, y_loc_train)
# info_gain=ig.getIG()
# print(sorted(info_gain,reverse=True))
# # print(iris.data)
# select_feature_name=ig.select_feature(X_loc_train,info_gain,0.95)
# X_loc_train = X_loc_train[select_feature_name].values
# X_loc_test = X_loc_test[select_feature_name].values


# t2=time.time()
# print('特征选择结束,耗时{}'.format(t2-t1))
# print('特征选择后shape',X_loc_train.shape )

# X_loc_train = X_loc_train.values
# X_loc_test = X_loc_test.values

if __name__=='__main__':

    best_estimators,best_params=get_best_params()
    model=lgb.LGBMClassifier(n_estimators=best_estimators, n_jobs=-1,
                       objective='binary', random_state=50, **best_params)

    # 模型部分
    # model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.02, n_estimators=2000,
    #                            max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
    #                            min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
    #                            colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)

    # model=XGBClassifier(
    #                         #eval_metric="auc",
    #                         # max_depth=8, learning_rate=0.05, n_estimators=2000, silent=True,
    #                         #      objective="binary:logistic", booster='gbtree', n_jobs=10, gamma=0, min_child_weight=5,
    #                         #      max_delta_step=0, subsample=0.8, colsample_bytree=1, colsample_bylevel=1,
    #                         #      reg_alpha=3, reg_lambda=5, scale_pos_weight=1, random_state=1000,
    #     max_depth=1, learning_rate=0.96, n_estimators=2000, silent=True,
    #     objective="binary:logistic", booster='gbtree', n_jobs=-1, gamma=0.48, min_child_weight=4,
    #      subsample=0.5, colsample_bytree=0.9,
    #     reg_alpha=0, reg_lambda=0.67,  random_state=1000,
    # )

    # 五折交叉训练，构造五个模型

    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=1024)

    # skf=list(StratifiedKFold(y_loc_train, n_folds=5, shuffle=True, random_state=1024))
    baseloss = []
    loss = 0

    dataset_blend_train = np.zeros((X_loc_train.shape[0], ))

    for i, (train_index, test_index) in enumerate(skf.split(X_loc_train,y_loc_train)):
        print("Fold", i)
        lgb_model = model.fit(X_loc_train[train_index], y_loc_train[train_index],
                              eval_names =['train','valid'],
                              eval_metric=tpr_weight_funtion_lc,
                              eval_set=[(X_loc_train[train_index], y_loc_train[train_index]),
                                        (X_loc_train[test_index], y_loc_train[test_index])],early_stopping_rounds=100
                              )
        baseloss.append(lgb_model.best_score_['valid']['TPR'])
        loss += lgb_model.best_score_['valid']['TPR']
        test_pred= lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]
        dataset_blend_train[test_index] = lgb_model.predict_proba(X_loc_train[test_index], num_iteration=lgb_model.best_iteration_)[:, 1]

        # xgb_model=model.fit(X_loc_train[train_index], y_loc_train[train_index],
        #
        #
        #                       eval_metric=tpr_weight_funtion_self,#evalmcc_min,#,函数似乎要加一个负号，才能正常运行；f最大，等价-f最小，应该是这样
        #                       eval_set=[(X_loc_train[train_index], y_loc_train[train_index]),
        #                                 (X_loc_train[test_index], y_loc_train[test_index])],
        #                     early_stopping_rounds=100
        #         )
        #
        # # print('xgb_model.evals_result',xgb_model.evals_result())
        # # print('xgb_model.best_score',xgb_model.best_score)#在验证集上的最好分数
        # # print('xgb_model.best_ntree_limit',xgb_model.best_ntree_limit)
        #
        #
        #
        # baseloss.append(xgb_model.best_score)
        # loss += xgb_model.best_score
        # test_pred = xgb_model.predict_proba(X_loc_test, ntree_limit=xgb_model.best_ntree_limit)[:, 1]
        # dataset_blend_train[test_index] = xgb_model.predict_proba(X_loc_train[test_index],
        #                                                           ntree_limit=xgb_model.best_ntree_limit)[:, 1]





        print('test mean:', test_pred.mean())
        res['prob_%s' % str(i)] = test_pred
    print('TPR:', baseloss, loss/5)

    # 加权平均
    res['Tag'] = 0
    for i in range(5):
        res['Tag'] += res['prob_%s' % str(i)]
    res['Tag'] = res['Tag']/5

    # 提交结果
    mean = res['Tag'].mean()
    print('mean:',mean)
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    res[['UID', 'Tag']].to_csv("lgb_baseline_%s_fs_yu.csv" % now, index=False)
    # pd.DataFrame(data=dataset_blend_train,columns=['val_yu']).to_csv('val_yu_{}_.csv'.format(now), index=False)


'''
开始特征选择
[0.3492115175481585, 0.32948921844531187, 0.3092562931395684, 0.3092562931395684, 0.30812670901542405, 0.30812670901542405, 0.2882465051961633, 0.28666628861006677, 0.26852420609331673, 0.263057930484761, 0.26016636105856833, 0.25442925240886344, 0.254139208173768, 0.25012121861939185, 0.24170274421916565, 0.24049088987571912, 0.2404440619557217, 0.23936130575157483, 0.23700594951566334, 0.23700594951566334, 0.23470695330601682, 0.2344169090709214, 0.22915802901040275, 0.2264149170938764, 0.22634808381501703, 0.21751819680783904, 0.21728365041281672, 0.21728365041281672, 0.21564760132525845, 0.2145180172011142, 0.20893221121888395, 0.20784306352974505, 0.20671347940560078, 0.20669261799102978, 0.2066257847121704, 0.2007840467518578, 0.19965446262771352, 0.19779589770499242, 0.19016994760756414, 0.18618025357501303, 0.18069966492138775, 0.17908041245804773, 0.17266646728983678, 0.17153688316569252, 0.1714764806231026, 0.1690061820932708, 0.15228814775518068, 0.15001686116922702, 0.1474784103337235, 0.13122517278310786, 0.12910847205265194, 0.1248255691613864, 0.12048999337221317, 0.11828942173502294, 0.10615944557253587, 0.10565845157389453, 0.10194002683231507, 0.10070985856739978, 0.09806441430690799, 0.09440274526276304, 0.09275828396029817, 0.08299442605265112, 0.08298795912216217, 0.08298795912216217, 0.07563789624540079, 0.07563789624540079, 0.07563789624540079, 0.07563789624540079, 0.07563789624540079, 0.07329510007695206, 0.06905618695093513, 0.06429191490141767, 0.04806894755382907, 0.045781451337436996, 0.02612358248008101, 0.02612358248008101, 0.02612358248008101, 0.025909992303959517, 0.025732303432998982, 0.02515165371503081, 0.025111122951203324, 0.024994269262913094, 0.024994269262913094]
原来特征： 13422
选择特征： 78
['code2_sum', 'code2', 'ip2_sum', 'ip2_sub_sum', 'ip2', 'ip2_sub', 'code1_sum', 'device1_x', 'code1', 'device2_x', 'device_code2_sum_y', 'mac1_sum_y', 'device_code1_sum_y', 'device1_y', 'device2_y', 'mac1_sum_x', 'device_code2_y', 'mac1_x', 'market_code_sum', 'market_type_sum', 'mac1_y', 'device_code1_y', 'geo_code_y', 'acc_id3_sum', 'acc_id2_sum', 'device_code3_sum_y', 'market_code', 'market_type', 'wifi_sum', 'wifi', 'version', 'device_code3_sum_x', 'device_code3_x', 'acc_id3', 'acc_id2', 'device_code2_sum_x', 'device_code2_x', 'device_code3_y', 'channel', 'geo_code_sum_y', 'device_code1_x', 'geo_code_x', 'mac2_sum', 'mac2', 'device_code1_sum_x', 'os', 'geo_code_sum_x', 'trans_type1', 'trans_type2', 'ip1_sub_y', 'ip1_y', 'amt_src2', 'amt_src1', 'device1_sum_y', 'mode', 'o_t_days', 'ip1_sub_x', 'merchant', 'ip1_x', 'acc_id1', 'acc_id1_sum', 'device2_sum_y', 'ip1_sum_y', 'ip1_sub_sum_y', 'day_y', 'channel_sum', 'amt_src1_sum', 'merchant_sum', 'trans_type1_sum', 'bal', 'amt_src2_sum', 'trans_type2_sum', 'device2_sum_x', 'trans_amt', 'day_x', 'mode_sum', 'os_sum', 'success']



0.9864440395988904
0.9859327447432762 #线上于线下较为一致
0.987405140968891#线下反而降了
0.9874752599638693#加了特征选择的分数，线上下降更厉害
0.9874770448438136 #去掉days_sum,days_sub,留下days_rate
0.9874951666699847#去掉days_sum,days_sub,days_rate //to

0.987945148966871#前一个基础上加所有变量

0.9876931129188804#信息增益0.95

0.9827681141776404#0.5

0.9866294587448905#0.7

0.987613563284944#0.75

0.9876931129188804#########0.8

#xgb
0.9883922

TPR
xgb [-0.811765, -0.738562, -0.701961, -0.688852, -0.796721] -0.7475721999999999
lgb: [-0.8032679738562092, -0.6826797385620915, -0.7147058823529412, -0.6931147540983607, -0.7986885245901639] -0.7384913746919534
   
   
TPR: [-0.819935, -0.722549, -0.721895, -0.682295, -0.786885] -0.7467117999999999#所有特征  
               
TPR: [-0.757843, -0.751961, -0.651634, -0.686885, -0.746557] -0.718976#basefeature
TPR: [-0.003922, -0.00719, -0.002941, -0.010164, -0.004918] -0.005827#crossfeature
TPR: [-0.003922, -0.00719, -0.002941, -0.010164, -0.004918] -0.005827#two_level_feature

#best params lgb
TPR: [-0.8179738562091503, -0.7970588235294118, -0.7607843137254902, -0.5449180327868852, -0.8170491803278688] -0.7475568413157612
#best params lgb,0.8 feature select
TPR: [-0.8274509803921568, -0.7879084967320262, -0.7584967320261438, -0.7304918032786886, -0.8144262295081968] -0.7837548483874425

#best params lgb,no early stopping  (一切和参数调优的条件是一样的，特征数量不变)
TPR: [-0.8372549019607842, -0.8035947712418301, -0.7957516339869282, -0.700327868852459, -0.7973770491803278] -0.7868612450444659

#best params lgb,0.8 feature select,no early stopping
TPR: [-0.8169934640522876, -0.7549019607843137, -0.7320261437908497, -0.720983606557377, -0.8075409836065575] -0.7664892317582771


#no early  new feature select
TPR: [-0.830718954248366, -0.7758169934640522, -0.7767973856209149, -0.7101639344262296, -0.8163934426229509] -0.7819781420765026

#new feature select,early stopping
TPR: [-0.8209150326797385, -0.7875816993464053, -0.765032679738562, -0.7491803278688525, -0.8131147540983606] -0.7871648987463838
'''