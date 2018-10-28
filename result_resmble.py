import pandas as pd
import datetime
from sklearn.linear_model import LogisticRegression

import gc
import os

# path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/kdxf/kdxf_data'
# train_file_name='round1_iflyad_train.txt'
# train1=pd.read_csv(os.path.join(path,train_file_name),delimiter='\t',
#                    usecols=['click'],
#                    # nrows=50
#                    )#dtype='category'#lightgbm能够直接处理'category'类型数据
#
#
#
#
#
# path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/kdxf/kdxf_data2'
# test_file_name='round2_iflyad_test_feature.txt'
# train_file_name='round2_iflyad_train.txt'
# train=pd.read_csv(os.path.join(path,train_file_name),delimiter='\t',
#                   # nrows=50,
#                   usecols=['click'],
#                   )#dtype='category'#lightgbm能够直接处理'category'类型数据
# test=pd.read_csv(os.path.join(path,test_file_name),delimiter='\t',
#                  # nrows=50
#                  usecols=['UID']
#                  )
# train=train.append(train1)
#
# print(test.shape)
#
# del train1
# gc.collect()
#
# y=train
# id =test
# predict_result=id
# print(y,id)

class resemble(object):
    def simple_add_weight(self):
        # (xi：0.55;yu:0.45)
        #(xi:0.45;yu:0.35,me:0.2)

        df1_xi=pd.read_csv('b1.csv')#lgb_baseline_09-29-16-16.csv
        df2_yu=pd.read_csv('b2.csv')
        # df3_me=pd.read_csv('lgb_baseline_09-30-15-28_me.csv')


        df=pd.merge(df1_xi,df2_yu,how='inner',on='UID')
        # df=pd.merge(df,df3_me,how='inner',on='UID')
        print(df.head())
        # df['Tag']=df[['Tag_x','Tag_y','Tag']].apply(
        #     lambda row:0.6*row['Tag_x']+0.3*row['Tag_y']+0.1*row['Tag'],axis=1)


        # print(df[['Tag_x','Tag_y']].head())
        df['Tag']=df[['Tag_x','Tag_y']].apply(
            lambda row:0.85*row['Tag_x']+0.15*row['Tag_y'],axis=1)
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M')
        df[['UID','Tag']].to_csv('ensmble_fs_{}.csv'.format(now),index=False)
        print(df.head())

    def harmonic_average(self):
        df1_xi = pd.read_csv('ensmble_fs_10-15-08-11.csv')  # lgb_baseline_09-29-16-16.csv
        df2_yu = pd.read_csv('ensmble_fs_10-15-14-19.csv')
        # df3_me=pd.read_csv('lgb_baseline_09-30-15-28_me.csv')


        df = pd.merge(df1_xi, df2_yu, how='inner', on='UID')
        # df=pd.merge(df,df3_me,how='inner',on='UID')
        print(df.head())
        # df['Tag']=df[['Tag_x','Tag_y','Tag']].apply(
        #     lambda row:0.6*row['Tag_x']+0.3*row['Tag_y']+0.1*row['Tag'],axis=1)


        # print(df[['Tag_x','Tag_y']].head())
        df['Tag'] = df[['Tag_x', 'Tag_y']].apply(
            lambda row: 2/(1/row['Tag_x'] +  1/row['Tag_y']), axis=1)
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M')
        df[['UID', 'Tag']].to_csv('ensmble_fs_{}.csv'.format(now), index=False)
        print(df.head())
        df1_xi=pd.read_csv('ensmble_fs_10-15-08-11.csv')#lgb_baseline_09-29-16-16.csv
        df2_yu=pd.read_csv('ensmble_fs_10-15-14-19.csv')
        # df3_me=pd.read_csv('lgb_baseline_09-30-15-28_me.csv')


        df=pd.merge(df1_xi,df2_yu,how='inner',on='UID')
        # df=pd.merge(df,df3_me,how='inner',on='UID')
        print(df.head())
        # df['Tag']=df[['Tag_x','Tag_y','Tag']].apply(
        #     lambda row:0.6*row['Tag_x']+0.3*row['Tag_y']+0.1*row['Tag'],axis=1)


        # print(df[['Tag_x','Tag_y']].head())
        df['Tag']=df[['Tag_x','Tag_y']].apply(
            lambda row:0.55*row['Tag_x']+0.45*row['Tag_y'],axis=1)
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M')
        df[['UID','Tag']].to_csv('ensmble_fs_harm{}.csv'.format(now),index=False)
        print(df.head())






    # def two_data_and_models_stacking(self):
    #     val_yu=pd.read_csv('val_yu_10-11-07-49_.csv')
    #     val_kd=pd.read_csv('val_kd_10-11-06-57_.csv')
    #     val_all=pd.concat([val_yu,val_kd],axis=1)
    #
    #     pred_yu=pd.read_csv('lgb_baseline_10-11-07-49_fs_yu.csv')['Tag']
    #     pred_kd=pd.read_csv('lgb_baseline_fs_10-11-06-57.csv')['Tag']
    #
    #     pred_all=pd.concat([pred_yu,pred_kd],axis=1)
    #     print('val_all.shape',val_all.shape)
    #     print('pred_all.shape',pred_all.shape)
    #     lr=LogisticRegression()
    #     lr.fit(val_all,y)
    #
    #     res=lr.predict_proba(pred_all)[:, 1]
    #     predict_result['Tag']=res
    #     print(predict_result.head())
    #     predict_result.to_csv('two_data_and_models_stacking.csv',index=False)
    #     pass
    # def avg_pred_add_weight(self):
    #     #lgb_clf,rf_clf
    #
    #     res=pd.read_csv('all_model_avg_pred_10-11-02-18.csv')
    #     res_add_weight=res[['lgb_clf','rf_clf']].apply(
    #         lambda row:0.85*row['lgb_clf']+0.15*row['rf_clf'],axis=1)
    #     predict_result['Tag'] = res_add_weight
    #     predict_result.to_csv('avg_pred_add_weight.csv', index=False)



if __name__=='__main__':
    resem=resemble()
    resem.simple_add_weight()
    # resem.harmonic_average()
    # resem.two_data_and_models_stacking()
    # resem.avg_pred_add_weight()
