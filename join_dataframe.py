import pandas as pd
#只有basefeature是带有tag可以直接使用的

df_train_base_feature=pd.read_csv('train_base_feature.csv')
df_train_cross_feature=pd.read_csv('train_cross_feature.csv')
df_test_base_feature=pd.read_csv('test_base_feature.csv')
df_test_cross_feature=pd.read_csv('test_cross_feature.csv')

df_train_two_level_feature=pd.read_csv('train_two_level_feature.csv')
df_test_two_level_feature=pd.read_csv('test_two_level_feature.csv')


train=pd.merge(df_train_base_feature,df_train_cross_feature,how='left',on='UID')
train=pd.merge(train,df_train_two_level_feature,how='left',on='UID')
# print('train.columns',train.columns)
train=train.drop(['days_rate'],axis=1)

test=pd.merge(df_test_base_feature,df_test_cross_feature,how='left',on='UID')
test=pd.merge(test,df_test_two_level_feature,how='left',on='UID')

test=test.drop(['days_rate'],axis=1)

train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)

'''
train.columns Index(['UID', 'Tag', 'day_x', 'success', 'mode', 'os', 'version', 'device1_x',
       'device2_x', 'device_code1_x', 'device_code2_x', 'device_code3_x',
       'mac1_x', 'mac2', 'ip1_x', 'ip2', 'wifi', 'geo_code_x', 'ip1_sub_x',
       'ip2_sub', 'day_y', 'trans_amt', 'bal', 'channel', 'amt_src1',
       'merchant', 'code1', 'code2', 'trans_type1', 'acc_id1',
       'device_code1_y', 'device_code2_y', 'device_code3_y', 'device1_y',
       'device2_y', 'mac1_y', 'ip1_y', 'amt_src2', 'acc_id2', 'acc_id3',
       'geo_code_y', 'trans_type2', 'market_code', 'market_type', 'ip1_sub_y',
       'days_sum', 'days_sub', 'days_rate', 'o_t_days'],
      dtype='object')
'''