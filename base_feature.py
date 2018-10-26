import pandas as pd
import sklearn.model_selection
import os
import numpy as np
path='/Users/ozintel/Downloads/Tsl_python_progect/local_ml/kaggle_competition/kaggle_competition_datas/sweet_orange/orange_data'
o_train=pd.read_csv(os.path.join(path,"operation_TRAIN.csv"),dtype=str)
t_train=pd.read_csv(os.path.join(path,"transaction_TRAIN.csv"),dtype=str)
o_test=pd.read_csv(os.path.join(path,"operation_round1.csv"),dtype=str)
t_test=pd.read_csv(os.path.join(path,"transaction_round1.csv"),dtype=str)
label_train=pd.read_csv(os.path.join(path,"tag_TRAIN.csv"),dtype=str)



#groupby("UID")
o_count_var=['day']+['mode','os','version','device1','device2','device_code1', 'device_code2', 'device_code3', 'mac1',
       'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']
o_sum_var=['success']#除了求和还要计算占比
o_nunique_var=['mode','os','version','device1','device2','device_code1', 'device_code2', 'device_code3', 'mac1',
       'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']

dict_o=dict([(each,'count') for each in o_count_var]+[(each,'sum') for each in o_sum_var]+[(each,'nunique') for each in o_nunique_var])



t_count_var=['day']+['channel',   'amt_src1', 'merchant',
       'code1', 'code2', 'trans_type1', 'acc_id1', 'device_code1',
       'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
        'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
       'market_code', 'market_type', 'ip1_sub']
# dict_t_count_var=dict([(each,'count') for each in t_count_var])

t_sum_var=['trans_amt','bal']
# dict_t_sum_var=dict([(each,'sum') for each in t_sum_var])

t_nunique_var=['channel',   'amt_src1', 'merchant',
       'code1', 'code2', 'trans_type1', 'acc_id1', 'device_code1',
       'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
        'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
       'market_code', 'market_type', 'ip1_sub']
# dict_t_nunique_var=dict([(each,'nunique') for each in t_nunique_var])


dict_t=dict([(each,'count') for each in t_count_var]+[(each,'sum') for each in t_sum_var]+[(each,'nunique') for each in t_nunique_var])


#
# #agg("UID")
# t_count=['channel',   'amt_src1', 'merchant',
#        'code1', 'code2', 'trans_type1', 'acc_id1', 'device_code1',
#        'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
#         'amt_src2', 'acc_id2', 'acc_id3', 'geo_code', 'trans_type2',
#        'market_code', 'market_type', 'ip1_sub']
#
# #有点说不通
# for each in t_count:
#     df_t_count=t_train.groupby(each)['UID'].agg('count').reset_index()




print('t_train')
t_train[t_sum_var]=t_train[t_sum_var].astype(np.float)
df_ttrain_user=t_train.groupby('UID').agg(dict_t).reset_index()
# df_ttrain_user).to_csv('t_train_user.csv',index=False)

print('o_train')
o_train[o_sum_var]=o_train[o_sum_var].astype(np.float)
df_otrain_user=o_train.groupby('UID').agg(dict_o).reset_index()
# df_otrain_user.to_csv('o_train_user.csv',index=False)

print('t_test')
t_test[t_sum_var]=t_test[t_sum_var].astype(np.float)
df_ttest_user=t_test.groupby('UID').agg(dict_t).reset_index()
# df_ttest_user.to_csv('t_test_user.csv',index=False)

print('o_test')
o_test[o_sum_var]=o_test[o_sum_var].astype(np.float)
df_otest_user=o_test.groupby('UID').agg(dict_o).reset_index()
# df_otest_user.to_csv('o_test_user.csv',index=False)




#同一个人groupby(字段)之后的操作次数、交易次数、同时进行


# o_train_user=pd.read_csv('o_train_user.csv')
# t_train_user=pd.read_csv('t_train_user.csv')

print('train')
train=pd.merge(label_train,df_otrain_user,how='left',on='UID')
train=pd.merge(train,df_ttrain_user,how='left',on='UID')
train.to_csv('train_base_feature.csv',index=False)

# o_test_user=pd.read_csv('o_test_user.csv')
# t_test_user=pd.read_csv('t_test_user.csv')
print('test')
test=pd.merge(df_otest_user,df_ttest_user,how='outer',on='UID')
test.to_csv('test_base_feature.csv',index=False)


