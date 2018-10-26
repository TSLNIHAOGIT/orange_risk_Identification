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

o_count_var=['day']
o_sum_var=['success']#除了求和还要计算占比
o_nunique_var=['mode','os','version','device1','device2','device_code1', 'device_code2', 'device_code3', 'mac1',
       'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']



t_count_var=['day']
# dict_t_count_var=dict([(each,'count') for each in t_count_var])

t_sum_var=['trans_amt','bal']
# dict_t_sum_var=dict([(each,'sum') for each in t_sum_var])

t_nunique_var=['channel',   'amt_src1', 'merchant',
       'code1', 'code2', 'trans_type1', 'acc_id1', 'device_code1',
       'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
        'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
       'market_code', 'market_type', 'ip1_sub']

#
df_label=label_train[['UID']]
#用户在不同字段下的操作次数交易次数
oo=o_sum_var+o_nunique_var
print('train oo')
for each in oo:
    df_temp=o_train.groupby(['UID',each])['day'].agg('count').reset_index()
    df_temp=df_temp.groupby('UID')['day'].agg([('{}_sum'.format(each),'sum')]).reset_index()
    df_label=pd.merge(df_label,df_temp,how='left',on='UID')

df_test=o_test[['UID']].drop_duplicates()
print('test oo')
for each in oo:
    df_temp=o_test.groupby(['UID',each])['day'].agg('count').reset_index()
    df_temp=df_temp.groupby('UID')['day'].agg([('{}_sum'.format(each),'sum')]).reset_index()
    df_test=pd.merge(df_test,df_temp,how='left',on='UID')


df2_label=label_train[['UID']]
df2_test=o_test[['UID']].drop_duplicates()##test id是不唯一的有很多

print('train tt')
for each in t_nunique_var:
    df_temp = t_train.groupby(['UID', each])['day'].agg('count').reset_index()
    df_temp = df_temp.groupby('UID')['day'].agg([('{}_sum'.format(each), 'sum')]).reset_index()
    df2_label = pd.merge(df2_label, df_temp, how='left', on='UID')

print('test tt')
for each in t_nunique_var:
    df_temp = t_test.groupby(['UID', each])['day'].agg('count').reset_index()
    df_temp = df_temp.groupby('UID')['day'].agg([('{}_sum'.format(each), 'sum')]).reset_index()
    df2_test = pd.merge(df2_test, df_temp, how='left', on='UID')



print('two_train merge')
two_train=pd.merge(df_label,df2_label,how='inner',on='UID')
two_train.to_csv('train_two_level_feature.csv',index=False)

print('two_test merge')
two_test=pd.merge(df_test,df2_test,how='outer',on='UID')
two_test.to_csv('test_two_level_feature.csv',index=False)










