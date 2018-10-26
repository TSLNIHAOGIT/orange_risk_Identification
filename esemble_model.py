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
from evaluation import tpr_weight_funtion_self,tpr_weight_funtion_lc,tpr_weight_funtion2

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


def model_train_predict( train_csr, train_y, predict_csr, predict_result):
    '''模型融合中使用到的各个单模型'''

    clfs = {
        'lgb_clf': lgb.LGBMClassifier(
                                      # metric='auc',
                                      boosting_type='gbdt', num_leaves=48, max_depth=-1,
                                      learning_rate=0.05, n_estimators=2000,
                                      max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                                      min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=1, reg_alpha=3, reg_lambda=5, random_state=1000, n_jobs=10,
                                      silent=True, ),

        'xgb_clf':xgb.XGBClassifier(
                            # eval_metric="auc",
                            max_depth=20, learning_rate=0.05,n_estimators=2000, silent=True,
                            objective="binary:logistic", booster='gbtree',n_jobs=10,  gamma=0, min_child_weight=5,
                            max_delta_step=0, subsample=0.8, colsample_bytree=1, colsample_bylevel=1,
                            reg_alpha=3, reg_lambda=5, scale_pos_weight=1,random_state=1000,),


        'et_gini_clf':ExtraTreesClassifier(n_estimators=2000, n_jobs=-1, criterion='gini'),
        'et_entory_clf':ExtraTreesClassifier(n_estimators=2000, n_jobs=-1, criterion='entropy'),

        'gbt_clf':GradientBoostingClassifier(learning_rate=0.05, subsample=0.9, max_depth=6, n_estimators=2000
                                             ,

                                             criterion='friedman_mse',
                                             init=None,
                                             loss='deviance',
                                             max_features=None, max_leaf_nodes=None,
                                             min_impurity_split=1e-07, min_samples_leaf=1,
                                             min_samples_split=2, min_weight_fraction_leaf=0.0,
                                              presort='auto', random_state=4,
                                              verbose=0, warm_start=False
                                             ),
        'bag_clf':BaggingClassifier(verbose=1,n_jobs=-1, n_estimators=2000, max_samples=1.0,
                                    max_features=1.0, bootstrap=True, bootstrap_features=False, random_state=1),
        ## 'ada_clf':AdaBoostClassifier(learning_rate=0.01,n_estimators=2000),#0.9603190970501208
        ## 'rf_clf': RandomForestClassifier(verbose=5, n_estimators=2000, max_depth=8, n_jobs=-1, criterion="gini"),#0.9778371414774663
        ##  'lr_clf' : LogisticRegression(n_jobs=-1,max_iter=1000),#效果太差
        ## 'svc_clf':SVC(probability=True,verbose=True),#速度太慢,效果差，比lr稍好
        ## 'gnb_clf':GaussianNB(),#有问题


    }


    '''切分一部分数据作为测试集'''
    # X, X_predict, y, y_predict = train_test_split(data, target, test_size=0.33, random_state=2017)
    # 元train中的一折预测结果组成的train，供第二层训练
    dataset_blend_train = np.zeros((train_csr.shape[0], len(clfs)))

    # 不带标签的预测结合的预测结果
    dataset_blend_test = np.zeros((predict_csr.shape[0], len(clfs)))

    '''5折stacking'''
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, random_state=2018, shuffle=True)
    val_scores=[]
    for j, clf_key in enumerate(clfs):
        clf = clfs[clf_key]
        '''依次训练各个单模型'''
        print(j, clf)
        dataset_blend_test_j = np.zeros((predict_csr.shape[0], n_folds))  # 临时使用
        for i, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
            '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
            print("Fold", i)
            X_train, y_train, X_test, y_test = train_csr[train_index], train_y[train_index], train_csr[test_index], \
                                               train_y[test_index]
            if clf_key!='gnb_clf':
                clf.set_params(random_state=i)
            if clf_key == 'lgb_clf':

                clf.fit(X_train, y_train,
                        eval_metric=tpr_weight_funtion_lc,
                        eval_set=[(X_train, y_train),
                                  (X_test, y_test)], early_stopping_rounds=100
                        )

                y_submission = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:, 1]
                # print('y_submission',y_submission)
                dataset_blend_train[test_index, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(predict_csr, num_iteration=clf.best_iteration_)[:,
                                             1]

            elif clf_key=='xgb_clf':

                clf.fit(X_train, y_train,
                        eval_metric=tpr_weight_funtion_self,
                        eval_set=[(X_train, y_train),
                                  (X_test, y_test)], early_stopping_rounds=100
                        )

                y_submission = clf.predict_proba(X_test, ntree_limit=clf.best_ntree_limit)[:, 1]
                # print('y_submission',y_submission)
                dataset_blend_train[test_index, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(predict_csr, ntree_limit=clf.best_ntree_limit)[:, 1]

                # self.best_iteration xgb
            else:
                clf.fit(X_train, y_train)
                y_submission = clf.predict_proba(X_test)[:, 1]
                # print('y_submission',y_submission)
                dataset_blend_train[test_index, j] = y_submission
                dataset_blend_test_j[:, i] = clf.predict_proba(predict_csr)[:, 1]

        '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(axis=1)  # （行取平均）k折预测取平均
        # auc_s=roc_auc_score(train_y, dataset_blend_train[:, j])
        tpr_s=tpr_weight_funtion2(train_y, dataset_blend_train[:, j])

        val_scores.append(tpr_s)
        print("val tpr Score: %f" % tpr_s)

    dd=dict(zip(clfs.keys(), val_scores))
    s_d = dict(sorted(dd.items(), key=lambda x: x[1]))
    print('val_scores:\n{}'.format(s_d))
    print('all model tpr mean:\n{}'.format(np.mean(val_scores)))
    print('模型开始融合训练')
    # now = datetime.datetime.now()
    # now = now.strftime('%m-%d-%H-%M')
    # pd.DataFrame(dataset_blend_test, columns=['lgb_clf', 'rf_clf']).to_csv('all_model_avg_pred_{}.csv'.format(now),
    #                                                                        index=False)
    #
    # lr = LogisticRegression(n_jobs=-1)
    # lr.fit(dataset_blend_train, train_y)
    # test_pred = lr.predict_proba(dataset_blend_test)[:, 1]

    gbc=GradientBoostingClassifier(learning_rate=0.05, subsample=0.9, max_depth=6, n_estimators=2000
                               ,

                               criterion='friedman_mse',
                               init=None,
                               loss='deviance',
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_split=1e-07, min_samples_leaf=1,
                               min_samples_split=2, min_weight_fraction_leaf=0.0,
                               presort='auto', random_state=4,
                               verbose=0, warm_start=False
                               )
    gbc.fit(dataset_blend_train, train_y)
    test_pred = gbc.predict_proba(dataset_blend_test)[:, 1]



    # svc_clf=SVC(probability=True)
    # svc_clf.fit(dataset_blend_train, train_y)
    # test_pred = svc_clf.predict_proba(dataset_blend_test)[:, 1]



    # # 模型部分
    # lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=5, max_depth=30, learning_rate=0.01,
    #                                n_estimators=1000,
    #                                max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
    #                                min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
    #                                colsample_bytree=1, reg_alpha=0.1, reg_lambda=0.1, seed=1000, n_jobs=10,
    #                                silent=True)
    #
    # lgb_model.fit(dataset_blend_train, train_y,eval_metric='auc',)
    # test_pred = lgb_model.predict_proba(dataset_blend_test, num_iteration=lgb_model.best_iteration_)[:, 1]

    predict_result['Tag'] = test_pred  ##########
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    predict_result[['UID', 'Tag']].to_csv("lgb_stacking%s.csv" % now, index=False)
    print(predict_result.head())

    # stack每次对train中的预测之后将预测结果放回相应的行,和原来的y对应

    ###################################


if __name__=='__main__':
    predict_result = test.loc[:, ['UID']]
    model_train_predict(X_loc_train, y_loc_train, X_loc_test, predict_result)


'''
['lgb_clf','xgb_clf','rf_clf','et_gini_clf','et_entory_clf','gbt_clf']
0.986720,0.987509,0.976579,0.982789,0.982689,0.985995

0.988024 0.988625，0.978081，0.983310，0.983547,0.986407#线上反而降了

0.988376 0.988924  0.977028 0.983158 0.983380 0.987063 //to

0.9851984872335061
[0.9887089722302679, 0.9894103300797533, 0.9777332569899733, 0.9836246299115132, 0.983780814295914, 0.9879329198936152]

val_scores:
[0.9887089722302679, 0.9894103300797533, 0.9833702834902018, 0.9838978769268151, 0.987665670446712, 0.9813493780576594, 0.9603190970501208, 0.9778371414774663]
all model auc mean:
0.9815698437198745

tpt
{'gbt_clf': -0.8138089005235601, 'et_gini_clf': -0.7894633507853404, 'et_entory_clf': -0.7893324607329841, 'xgb_clf': -0.781282722513089, 'bag_clf': -0.7615837696335077, 'lgb_clf': -0.7583769633507853, 'rf_clf': -0.6467277486910994, 'ada_clf': -0.42755235602094244, 'svc_clf': -0.26747382198952885, 'lr_clf': -0.06426701570680629}
all model tpr mean:
-0.6099869109947644


{'gbt_clf': -0.8143979057591623, 'et_entory_clf': -0.7893324607329844, 'et_gini_clf': -0.7865837696335078, 'lgb_clf': -0.7683246073298429, 'xgb_clf': -0.762303664921466, 'bag_clf': -0.7568717277486912}
all model tpr mean:
-0.7796356893542757

'''

