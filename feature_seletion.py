from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np
import math
import pandas as pd
iris=load_iris()
# #递归特征消除法，返回特征选择后的数据
# #参数estimator为基模型
# #参数n_features_to_select为选择的特征个数
# feature_select=RFE(estimator=LogisticRegression(), n_features_to_select=2)
# # feature_select = RFE(estimator=lgb_model, n_features_to_select=int(train_csr.shape[1] * 0.95))
# # feature_select = SelectPercentile(chi2,percentile=percent)#原来是95  #同一个类别变量既有one-hot,又有label-encoder
# feature_select.fit(iris.data, iris.target)
#
# train_csr = feature_select.transform(iris.data)
# print(train_csr[0:100])
# print('train_csr 已经完毕')

# predict_csr = feature_select.transform(predict_csr)




"""
信息增益进行特征选择（嫁接来的）
"""
class IG():
    def __init__(self,X,y):
        X = np.array(X)
        n_feature = np.shape(X)[1]
        n_y = len(y)

        orig_H = 0
        for i in set(y):
            orig_H += -(y.tolist().count(i)/n_y)*math.log(y.tolist().count(i)/n_y)

        condi_H_list = []
        for i in range(n_feature):
            feature = X[:,i]
            sourted_feature = sorted(feature)
            threshold = [(sourted_feature[inde-1]+sourted_feature[inde])/2 for inde in range(len(feature)) if inde != 0 ]

            if max(feature) in threshold:
                threshold.remove(max(feature))
            if min(feature) in threshold:
                threshold.remove(min(feature))

            pre_H = 0
            for thre in set(threshold):
                lower = [y[s] for s in range(len(feature)) if feature[s] < thre]
                highter = [y[s] for s in range(len(feature)) if feature[s] > thre]
                H_l = 0
                for l in set(lower):
                    H_l += -(lower.count(l) / len(lower))*math.log(lower.count(l) / len(lower))
                H_h = 0
                for h in set(highter):
                    H_h += -(highter.count(h) / len(highter))*math.log(highter.count(h) / len(highter))
                temp_condi_H = len(lower)/n_y *H_l+ len(highter)/n_y * H_h
                condi_H = orig_H - temp_condi_H
                pre_H = max(pre_H,condi_H)
            condi_H_list.append(pre_H)

        self.IG = condi_H_list


    def getIG(self):
        return self.IG

    def select_feature(self,all_feature, information_gain, select_percentile):
        """
        信息增益选择特征
        """
        feature_ig = {}  # 特征-信息增益键值对
        print("原来特征：", len(all_feature))
        for index, x in enumerate(all_feature):#为dataframe的列名称

            # print('x',x)
            feature_ig[x] = information_gain[index]

        # print('feature_ig',feature_ig)

        sorted_feature = sorted(feature_ig.items(), key=lambda x: x[1], reverse=True)  # 从大到小
        feature_select = list()
        for x in sorted_feature:
            feature_select.append(x[0])
        feature_select = feature_select[:int(len(feature_select) * select_percentile)]
        print("选择特征：", len(feature_select))
        print(feature_select)

        return feature_select

if __name__=='__main__':
    ig=IG(iris.data, iris.target)
    info_gain=ig.getIG()
    print(info_gain)
    # print(iris.data)
    ig.select_feature(pd.DataFrame(iris.data,columns=['v1','v2','v3','v4']),info_gain,0.95)