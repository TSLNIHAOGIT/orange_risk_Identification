import pandas as pd
import sklearn.model_selection
import os
import numpy as np
import pandas as pd
path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/sweet_orange/orange_data'
o_train=pd.read_csv(os.path.join(path,"operation_TRAIN.csv"),dtype=str)
t_train=pd.read_csv(os.path.join(path,"transaction_TRAIN.csv"),dtype=str)
o_test=pd.read_csv(os.path.join(path,"operation_round1.csv"),dtype=str)
t_test=pd.read_csv(os.path.join(path,"transaction_round1.csv"),dtype=str)
label_train=pd.read_csv(os.path.join(path,"tag_TRAIN.csv"),dtype=str)

#同一个人操作与交易天数之和、之差、同一天进行操作和交易的次数（隔一天进行交易的次数）

t_count_var=['day']
dict_t_count_var=dict([(each,'count') for each in t_count_var])


df_day_otrain=o_train.groupby('UID').agg(dict_t_count_var).reset_index()
df_day_ttrain=t_train.groupby('UID').agg(dict_t_count_var).reset_index()
df_sum_days=pd.merge(df_day_otrain,df_day_ttrain,how='inner',on='UID')
df_sum_days['days_sum']=df_sum_days.apply(lambda row:row['day_x']+row['day_y'],axis=1)
df_sum_days['days_sub']=df_sum_days.apply(lambda row:row['day_x']-row['day_y'],axis=1)
df_sum_days['days_rate']=df_sum_days.apply(lambda row:row['day_x']/row['day_y'],axis=1)
df_sum_days=df_sum_days.drop(['day_x','day_y'],axis=1)


#有人只进行操作有人只进行交易，有人两个都进行了

dic_oday=dict(list(o_train[['UID','day']].groupby('UID')))
dic_tday=dict(list(t_train[['UID','day']].groupby('UID')))
keys=set(dic_tday.keys())&set(dic_oday.keys())
alls=[]
for key in keys:
    d={}
    # print('key',key)
    d['UID']=key
    d['o_t_days']=\
    len(set(dic_tday[key]['day'])&set(
    dic_oday[key]['day']))
    alls.append(d)
o_t_days=pd.DataFrame(alls)


df1=pd.merge(label_train[["UID"]],df_sum_days,how='left',on='UID')
df2=pd.merge(df1,o_t_days,how='left',on='UID')
df2.to_csv('train_cross_feature.csv',index=False)


df_day_otest=o_test.groupby('UID').agg(dict_t_count_var).reset_index()
df_day_ttest=t_test.groupby('UID').agg(dict_t_count_var).reset_index()
dft_sum_days=pd.merge(df_day_otest,df_day_ttest,how='inner',on='UID')
dft_sum_days['days_sum']=dft_sum_days.apply(lambda row:row['day_x']+row['day_y'],axis=1)
dft_sum_days['days_sub']=dft_sum_days.apply(lambda row:row['day_x']-row['day_y'],axis=1)
dft_sum_days['days_rate']=dft_sum_days.apply(lambda row:row['day_x']/row['day_y'],axis=1)
dft_sum_days=dft_sum_days.drop(['day_x','day_y'],axis=1)


dic_t_oday=dict(list(o_test[['UID','day']].groupby('UID')))
dic_t_tday=dict(list(t_test[['UID','day']].groupby('UID')))
keys=set(dic_t_tday.keys())&set(dic_t_oday.keys())
alls_t=[]
for key in keys:
    d={}
    # print('key',key)
    d['UID']=key
    d['o_t_days']=\
    len(set(dic_t_tday[key]['day'])&set(
    dic_t_oday[key]['day']))
    alls_t.append(d)
o_t_days=pd.DataFrame(alls_t)


dft2=pd.merge(dft_sum_days,o_t_days,how='outer',on='UID')
dft2.to_csv('test_cross_feature.csv',index=False)
