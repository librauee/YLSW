# 导入相关库
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
from datetime import datetime
import lightgbm as lgb

warnings.filterwarnings('ignore')

# 读取商户静态信息
print("*******************************")
print("开始读取商户静态信息")
rename_col_list = ['BID', 'YLID', 'BSID', 'BINFO', 'YLDATE', 'PROVINCE', 'CITY', 'COUNTY', 'RANK']
train_base = pd.read_csv("train_mchnt_1.csv", header=None)
train_base.columns = rename_col_list + ['label']
# t为不同训练、测试集标志位
train_base['t'] = 1
test_base1 = pd.read_csv("final_data_set/train_mchnt_2.csv", header=None)
test_base1.columns = rename_col_list + ['label']
test_base1['t'] = 2
test_base2 = pd.read_csv("final_data_set/test_mchnt_2.csv", header=None)
test_base2.columns = rename_col_list
test_base2['t'] = 3

# 合并训练集与测试集
base = pd.concat([train_base, test_base1, test_base2], axis=0)

print("开始处理静态信息特征")
base['VIP_flag'] = base['RANK'].apply(lambda x: 1 if 11 <= x <= 16 else 0)

# 类别变量计数统计特征
for col in ['YLID', 'BSID', 'BINFO', 'PROVINCE', 'CITY', 'COUNTY', 'RANK']:
    base[col + '_base_count'] = base[col].map(base[col].value_counts())
    col_idx = base[col].value_counts()
    for idx in col_idx[col_idx < 10].index:
        base[col] = base[col].replace(idx, -1)

# 加入银联商务服务日期特征
base['YLDATE'] = base['YLDATE'].astype('datetime64')
base['year'] = base['YLDATE'].apply(lambda x:x.year)
base['month'] = base['YLDATE'].apply(lambda x:x.month)
base['age'] = 2021 - base['year']
base['month'] = base.apply(lambda x: (x['age'] - 1) * 12 - x['month'] + 7 if x['t'] == 1 else (x['age'] - 1) * 12 - x['month'] + 8, axis=1)
base['weekday'] = base['YLDATE'].apply(lambda x:x.weekday())
base['weekday_flag'] = base['weekday'].apply(lambda x: 1 if x == 6 or x == 5 else 0)
del base['weekday']
new_data = base.copy()
print("完成静态信息特征处理")
print("*******************************")

# 读取交易流水信息
print("开始读取交易流水信息")
rename_col_list = ['BID', 'TTIME', 'CARD', 'LID1', 'LID2', 'AMOUNT', 'MODE', 'FEE1', 'FEE2', 'CARD_ADDR', 'IND', 'PID',
                   'TID', 'BAPP']
train = pd.read_csv("train_trans_1.csv", header=None)
train.columns = rename_col_list

test1 = pd.read_csv("test_trans_1.csv", header=None)
test1.columns = rename_col_list

test2 = pd.read_csv("final_data_set/test_trans_2.csv", header=None)
test2.columns = rename_col_list
# TYPE为不同训练、测试集标志位
train['TYPE'] = 'train'
test1['TYPE'] = 'test1'
test2['TYPE'] = 'test2'

# 合并训练集与测试集
data = pd.concat([train, test1, test2], axis=0)

# 月、日、时交易时间特征分离
data['TTIME_M'] = data['TTIME'].apply(lambda x: int(str(x)[0]))
data['TTIME_D'] = data['TTIME'].apply(lambda x: int(str(x)[1:3]))
data['TTIME_H'] = data['TTIME'].apply(lambda x: int(str(x)[3:5]))

print("正在清洗数据，删除异常值")
# 删除异常数据
data_1 = data[((data['TTIME_M'] == 4) | (data['TTIME_M'] == 5) | (data['TTIME_M'] == 6)) & (data['TYPE'] == 'train')]
data_2 = data[((data['TTIME_M'] == 5) | (data['TTIME_M'] == 6) | (data['TTIME_M'] == 7)) & (data['TYPE'] != 'train')]
data = pd.concat([data_1, data_2], axis=0)
del data_1, data_2

# 判断银行卡类别, 1为信用卡
def judge_type(x):
    if math.isnan(x['LID2']):
        if x['LID1'] == 5 or x['LID1'] == 3:
            return 1
        else:
            return 0
    if 2 <= int(x['LID2']) <= 3:
        return 1
    else:
        return 0


data['CARD_TYPE'] = data.apply(judge_type, axis=1)
data.sort_values(['TYPE', 'BID', 'TTIME'], inplace=True)
data.reset_index(inplace=True, drop=True)

print("正在处理缺失值")
# 缺失值填充
nunique_f = ['CARD', 'LID1', 'LID2', 'MODE', 'CARD_ADDR', 'IND', 'PID', 'TID', 'BAPP', 'CARD_TYPE']
data[nunique_f] = data[nunique_f].fillna('-999')


print("开始处理交易流水特征")
print("开始处理商户交易日期间隔总特征")
# 商户交易日期间隔特征（总）
# 均值、方差、最大值、最小值等
train_end_date = '2020-07-01'
train_end_date = datetime.strptime(train_end_date, '%Y-%m-%d')
train_end_day = train_end_date.timetuple().tm_yday
test_end_date = '2020-08-01'
test_end_date = datetime.strptime(test_end_date, '%Y-%m-%d')
test_end_day = test_end_date.timetuple().tm_yday
data['T_'] = data['TTIME'].apply(lambda x: '2020-0' + str(x)[0] + '-' + str(x)[1:3])
data['T_'] = pd.to_datetime(data['T_'])
data['T_'] = data['T_'].apply(lambda x: x.dayofyear)
data['D_diff'] = data.apply(lambda x: train_end_day - x['T_'] if x['TYPE'] == 'train' else test_end_day - x['T_'],
                            axis=1)

df_temp = data.groupby(['BID'])['D_diff'].agg([
    ('day_max', lambda x: x.max()),
    ('day_min', lambda x: x.min()),
    ('day_max-min', lambda x: x.max() - x.min()),
])
new_data = pd.merge(new_data, df_temp, on='BID', how='left')
df_temp = data.groupby(['BID'])['T_'].agg([
    ('day_range_max', lambda x: x.diff().max()),
    ('day_range_min', lambda x: x.diff().min()),
    ('day_range_mean', lambda x: x.diff().mean()),
    ('day_range_std', lambda x: x.diff().std()),
    ('day_range_skew', lambda x: x.diff().skew()),
])
new_data = pd.merge(new_data, df_temp, on='BID', how='left')

print("开始处理商户交易金额、费用总特征")
# 交易金额、费用特征总计
# 均值、方差、最大值、最小值、和、相邻交易最大间隔
amount_f = ['AMOUNT', 'FEE1', 'FEE2']
for f in amount_f:
    df_temp = data.groupby('BID')[f].agg([
        ('{}_mean'.format(f), 'mean'),
        ('{}_std'.format(f), 'std'),
        ('{}_max'.format(f), 'max'),
        ('{}_min'.format(f), 'min'),
        ('{}_sum'.format(f), 'sum'),
        ('{}_range_max'.format(f), lambda x: x.diff().max()),
        ('{}_range_min'.format(f), lambda x: x.diff().min()),
    ]).reset_index()
    new_data = pd.merge(new_data, df_temp, on='BID', how='left')

train = data[data['TYPE'] == 'train']
train.reset_index(inplace=True, drop=True)
test = data[data['TYPE'] != 'train']
test.reset_index(inplace=True, drop=True)
# 按月组合统计特征

print("开始处理商户交易金额、费用月特征")
for f in amount_f:
    tmp_train = train.groupby(['BID', 'TTIME_M'])[f].agg([
        ('{}_mean'.format(f), 'mean'),
        ('{}_std'.format(f), 'std'),
        ('{}_max'.format(f), 'max'),
        ('{}_min'.format(f), 'min'),
        ('{}_sum'.format(f), 'sum'),
        ('{}_range_max'.format(f), lambda x: x.diff().max()),
        ('{}_range_min'.format(f), lambda x: x.diff().min()),
    ]).reset_index()
    tmp_test = test.groupby(['BID', 'TTIME_M'])[f].agg([
        ('{}_mean'.format(f), 'mean'),
        ('{}_std'.format(f), 'std'),
        ('{}_max'.format(f), 'max'),
        ('{}_min'.format(f), 'min'),
        ('{}_sum'.format(f), 'sum'),
        ('{}_range_max'.format(f), lambda x: x.diff().max()),
        ('{}_range_min'.format(f), lambda x: x.diff().min()),
    ]).reset_index()
    for agg_type in tqdm(['mean', 'std', 'max', 'min', 'sum', 'range_max', 'range_min']):
        tmp_train_ = tmp_train.pivot_table(index='BID', columns='TTIME_M', values=['{}_{}'.format(f, agg_type)])[
            '{}_{}'.format(f, agg_type)][[4, 5, 6]]
        tmp_train_.columns = ['{}_{}_{}'.format(i, f, agg_type) for i in range(3)]
        tmp_test_ = tmp_test.pivot_table(index='BID', columns='TTIME_M', values=['{}_{}'.format(f, agg_type)])[
            '{}_{}'.format(f, agg_type)][[5, 6, 7]]
        tmp_test_.columns = ['{}_{}_{}'.format(i, f, agg_type) for i in range(3)]
        tmp = pd.concat([tmp_train_, tmp_test_], axis=0)
        new_data = pd.merge(new_data, tmp, on='BID', how='left')

# 金钱不同月做差，求比例
for f in tqdm(amount_f):
    for agg_type in ['mean', 'sum']:
        new_data[f'{f}_{agg_type}_diff1'] = new_data[f'2_{f}_{agg_type}'] - new_data[f'0_{f}_{agg_type}']
        new_data[f'{f}_{agg_type}_diff2'] = new_data[f'2_{f}_{agg_type}'] - new_data[f'1_{f}_{agg_type}']
        new_data[f'{f}_{agg_type}_diff3'] = new_data[f'1_{f}_{agg_type}'] - new_data[f'0_{f}_{agg_type}']
    new_data[f'{f}_sum_divide'] = new_data[f'2_{f}_sum'] / (new_data[f'{f}_sum'] + 1e-8)

print("开始统计滑动窗口特征")
# 最长连续交易天数
def longestConsecutive(nums):
    if not nums:
        return 0
    nums = list(set(nums))
    nums.sort()
    res, i, nums_len = 1, 0, len(nums)
    while i < nums_len - 1:
        tmp = 1
        while i < nums_len - 1 and nums[i] == nums[i + 1] - 1:
            tmp += 1
            i += 1
        res = max(res, tmp)
        i += 1
    return res


df_temp = data.groupby('BID')['T_'].apply(lambda x: x.tolist())
new_data = pd.merge(new_data, df_temp, on='BID', how='left')
new_data['longest'] = new_data['T_'].apply(longestConsecutive)
del new_data['T_']

# 滑动窗口特征统计
data['last_week_0'] = data['D_diff'].apply(lambda x: 1 if x <= 7 else 0)
data['last_week_1'] = data['D_diff'].apply(lambda x: 1 if x <= 14 else 0)
data['last_week_2'] = data['D_diff'].apply(lambda x: 1 if x <= 21 else 0)

week_windows = ['last_week_0', 'last_week_1', 'last_week_2']
for window in tqdm(week_windows):
    df_temp = data[data[window] == 1]
    for f in nunique_f:
        temp2 = df_temp.groupby(['BID'])[f].agg([
            ('{}_{}_count'.format(window, f), 'count'),
            ('{}_{}_nunique'.format(window, f), 'nunique'),
        ]).reset_index()
        new_data = pd.merge(new_data, temp2, on='BID', how='left')
    for f in amount_f:
        temp2 = df_temp.groupby(['BID'])[f].agg([
            ('{}_{}_mean'.format(window, f), 'mean'),
            ('{}_{}_std'.format(window, f), 'std'),
            ('{}_{}_max'.format(window, f), 'max'),
            ('{}_{}_min'.format(window, f), 'min'),
        ]).reset_index()
        new_data = pd.merge(new_data, temp2, on='BID', how='left')

for f in tqdm(nunique_f):
    new_data['week_0_1_ratio_{}_count'.format(f)] = new_data['last_week_0_{}_count'.format(f)] / new_data[
        'last_week_1_{}_count'.format(f)]
    new_data['week_0_2_ratio_{}_count'.format(f)] = new_data['last_week_0_{}_count'.format(f)] / new_data[
        'last_week_2_{}_count'.format(f)]
    new_data['week_0_1_ratio_{}_nunique'.format(f)] = new_data['last_week_0_{}_nunique'.format(f)] / new_data[
        'last_week_1_{}_nunique'.format(f)]
    new_data['week_0_2_ratio_{}_nunique'.format(f)] = new_data['last_week_0_{}_nunique'.format(f)] / new_data[
        'last_week_2_{}_nunique'.format(f)]
for agg_type in tqdm(['max', 'min', 'mean']):
    for f in amount_f:
        new_data['week_0_1_ratio_{}_{}'.format(f, agg_type)] = new_data['last_week_0_{}_{}'.format(f, agg_type)] / \
                                                               new_data['last_week_1_{}_{}'.format(f, agg_type)]
        new_data['week_0_2_ratio_{}_{}'.format(f, agg_type)] = new_data['last_week_0_{}_{}'.format(f, agg_type)] / \
                                                               new_data['last_week_2_{}_{}'.format(f, agg_type)]

print("开始处理商户交易日期间隔月特征")
# 商户交易日期间隔特征（月）
tmp_train = train.groupby(['BID', 'TTIME_M'])['TTIME_D'].agg([
    ('day_range_max', lambda x: x.diff().max()),
    ('day_range_min', lambda x: x.diff().min()),
    ('day_range_mean', lambda x: x.diff().mean()),
    ('day_range_std', lambda x: x.diff().std()),
    ('time_count', 'count'),
    ('time_nunique', 'nunique'),
   ]).reset_index()
tmp_test = test.groupby(['BID', 'TTIME_M'])['TTIME_D'].agg([
    ('day_range_max', lambda x: x.diff().max()),
    ('day_range_min', lambda x: x.diff().min()),
    ('day_range_mean', lambda x: x.diff().mean()),
    ('day_range_std', lambda x: x.diff().std()),
    ('time_count', 'count'),
    ('time_nunique', 'nunique'),
   ]).reset_index()
for agg_type in tqdm(['day_range_max', 'day_range_min', 'day_range_mean', 'day_range_std', 'time_count', 'time_nunique']):
    tmp_train_ = tmp_train.pivot_table(index='BID', columns='TTIME_M', values=[agg_type]).loc[:, [(f'{agg_type}', 4), (f'{agg_type}', 5), (f'{agg_type}', 6)]]
    tmp_train_.columns = ['{}_{}'.format(i, agg_type) for i in range(3)]
    tmp_test_ = tmp_test.pivot_table(index='BID', columns='TTIME_M', values=[agg_type]).loc[: ,[(f'{agg_type}', 5), (f'{agg_type}', 6), (f'{agg_type}', 7)]]
    tmp_test_.columns = ['{}_{}'.format(i, agg_type) for i in range(3)]
    tmp = pd.concat([tmp_train_, tmp_test_], axis=0)
    new_data = pd.merge(new_data, tmp, on='BID', how='left')

print("开始处理商户月间相关特征")
# diff特征
# 不同月份特征做差
for agg_type in tqdm(['day_range_mean', 'time_count', 'time_nunique']):
    new_data[f'{agg_type}_sum'] = new_data[f'0_{agg_type}'] + new_data[f'1_{agg_type}'] + new_data[f'2_{agg_type}']
    new_data[f'{agg_type}_divide'] = new_data[f'2_{agg_type}'] / (new_data[f'{agg_type}_sum'] + 1e-8)
for agg_type in tqdm(['day_range_max', 'day_range_min', 'day_range_mean', 'time_count', 'time_nunique']):
    new_data[f'{agg_type}_diff1'] = new_data[f'2_{agg_type}'] - new_data[f'0_{agg_type}']
    new_data[f'{agg_type}_diff2'] = new_data[f'2_{agg_type}'] - new_data[f'1_{agg_type}']
    new_data[f'{agg_type}_diff3'] = new_data[f'1_{agg_type}'] - new_data[f'0_{agg_type}']
for i in range(3):
    new_data[f'{i}_time_mean'] = new_data[f'{i}_time_count'] / (new_data[f'{i}_time_nunique'] + 1e-8)

# unique特征
for f in tqdm(nunique_f):
    tmp_train = train.groupby(['BID', 'TTIME_M'])[f].agg([
    ('{}_N'.format(f), 'nunique'),
    ]).reset_index()
    tmp_test = test.groupby(['BID', 'TTIME_M'])[f].agg([
    ('{}_N'.format(f), 'nunique'),
    ]).reset_index()
    agg_type = '{}_N'.format(f)
    tmp_train_ = tmp_train.pivot_table(index='BID', columns='TTIME_M', values=[agg_type]).loc[:, [(f'{agg_type}', 4), (f'{agg_type}', 5), (f'{agg_type}', 6)]]
    tmp_train_.columns = ['{}_{}'.format(i, agg_type) for i in range(3)]
    tmp_test_ = tmp_test.pivot_table(index='BID', columns='TTIME_M', values=[agg_type]).loc[: ,[(f'{agg_type}', 5), (f'{agg_type}', 6), (f'{agg_type}', 7)]]
    tmp_test_.columns = ['{}_{}'.format(i, agg_type) for i in range(3)]
    tmp = pd.concat([tmp_train_, tmp_test_], axis=0)
    new_data = pd.merge(new_data, tmp, on='BID', how='left')
    new_data[f'{agg_type}_unique_diff1'] = new_data[f'2_{agg_type}'] - new_data[f'0_{agg_type}']
    new_data[f'{agg_type}_unique_diff2'] = new_data[f'2_{agg_type}'] - new_data[f'1_{agg_type}']
    new_data[f'{agg_type}_unique_diff3'] = new_data[f'1_{agg_type}'] - new_data[f'0_{agg_type}']
    new_data[f'{agg_type}_unique_sum'] = new_data[f'0_{agg_type}'] + new_data[f'1_{agg_type}'] + new_data[f'2_{agg_type}']
    new_data[f'{agg_type}_unique_divide'] = new_data[f'2_{agg_type}'] / (new_data[f'{agg_type}_unique_sum'] + 1e-8)

print("开始处理小时级别的商户交易金额特征")
# 按小时做交易金额统计
f = 'AMOUNT'
df_temp = data.groupby(['BID', 'TTIME_H'])[f].agg([
      ('{}_N'.format(f), 'nunique'),
      ('{}_count'.format(f), 'count'),
      ('{}_mean'.format(f), 'mean') ,
      ('{}_std'.format(f), 'std'),
      ('{}_max'.format(f), 'max'),
      ('{}_min'.format(f), 'min'),
    ]).reset_index()
agg_types = ['N', 'count', 'mean']
df_temp = df_temp.pivot_table(index='BID', columns='TTIME_H', values=['{}_{}'.format(f, agg_type) for agg_type in agg_types],fill_value=0)
df_temp.columns = ['{}_{}_{}'.format(f, agg_type,j) for agg_type in agg_types for j in range(24)]
new_data = pd.merge(new_data, df_temp, on='BID', how='left')
print("已经完成商户交易流水特征处理")
print("*******************************")

# 模型一
# 将初赛的训练集与测试集当成复赛的训练集使用
print("划分训练集与测试集")
X_train = new_data[new_data['t'] != 3]
X_test = new_data[new_data['t'] == 3]
X_test['t'] = np.nan
y = X_train['label']

drop_cols = ['BID', 'YLDATE']
cat_cols = ['YLID', 'BSID', 'BINFO', 'PROVINCE', 'CITY', 'COUNTY', 'year']

X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)
X_train[cat_cols] = X_train[cat_cols].astype('category')
X_test[cat_cols] = X_test[cat_cols].astype('category')
features = X_train.columns
features = features.drop('label')
print("原始特征共计{}维".format(len(features)))
print("使用树模型进行特征筛选")
# 模型训练、预测
KF = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
X_train[cat_cols] = X_train[cat_cols].astype('category')
X_test[cat_cols] = X_test[cat_cols].astype('category')
params = {
          'objective':'binary',
          'metric':'binary_error',
          'learning_rate':0.05,
          'subsample':0.8,
          'subsample_freq':3,
          'colsample_btree':0.8,
          'num_iterations': 10000,
          'silent':True
}
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros((len(X_test)))
# 特征重要性
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})
# 五折交叉验证
for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
    print("fold n°{}".format(fold_))
    print('trn_idx:',trn_idx)
    print('val_idx:',val_idx)
    trn_data = lgb.Dataset(X_train.iloc[trn_idx][features],label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx][features],label=y.iloc[val_idx])
    num_round = 10000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets = [trn_data, val_data],
        verbose_eval=500,
        early_stopping_rounds=200,
        categorical_feature=cat_cols,
    )
    feat_imp_df['imp'] += clf.feature_importance() / 5
    oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration)
print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))


drop_f = feat_imp_df[feat_imp_df['imp'] < 0.2]['feat']
for f in drop_f:
    if f in cat_cols:
        cat_cols.remove(f)
features = features.drop(drop_f)

print("得到最终特征共计{}维".format(len(features)))

print("开始模型一的训练与预测")
# 模型训练、预测
KF = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
X_train[cat_cols] = X_train[cat_cols].astype('category')
X_test[cat_cols] = X_test[cat_cols].astype('category')
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.05,
    'subsample': 0.8,
    'subsample_freq': 3,
    'colsample_btree': 0.8,
    'num_iterations': 10000,
    'silent': True
}
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros((len(X_test)))

# 五折交叉验证
for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
    print("fold n°{}".format(fold_))
    print('trn_idx:', trn_idx)
    print('val_idx:', val_idx)
    trn_data = lgb.Dataset(X_train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx][features], label=y.iloc[val_idx])
    num_round = 10000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets=[trn_data, val_data],
        verbose_eval=500,
        early_stopping_rounds=200,
        categorical_feature=cat_cols,
    )

    oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration)
print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

test = new_data[new_data['t'] == 3][['BID']]
test['predict'] = predictions_lgb / 5
test.to_csv('lgb_predict.csv', index=False)
print("完成模型一的训练与预测")
print("*******************************")

print("Null importance特征筛选")
X_train = new_data[new_data['t'] == 2]
X_train_ = new_data[new_data['t'] == 1]
X_test = new_data[new_data['t'] == 3]
X_test['t'] = np.nan
y = X_train['label']
y_ = X_train_['label']

drop_cols = ['BID', 'YLDATE']
cat_cols = ['YLID', 'BSID', 'BINFO', 'PROVINCE', 'CITY', 'COUNTY', 'year']

X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)
X_train_.drop(drop_cols, axis=1, inplace=True)
X_train[cat_cols] = X_train[cat_cols].astype('category')
X_test[cat_cols] = X_test[cat_cols].astype('category')
X_train_[cat_cols] = X_train_[cat_cols].astype('category')

features = X_train.columns
features = features.drop('label')


def get_feature_importances(data, shuffle, seed=None):
    train_features = [f for f in data if f not in ['label']]
    y = data['label'].copy()
    if shuffle:
        y = data['label'].copy().sample(frac=1.0)
    dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'rf',
        'subsample': 0.623,
        'colsample_bytree': 0.7,
        'num_leaves': 127,
        'max_depth': 8,
        'seed': seed,
        'bagging_freq': 1,
        'n_jobs': 4
    }
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200, categorical_feature=cat_cols)
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))

    return imp_df


actual_imp_df = get_feature_importances(pd.concat([X_train, X_train_], axis=0), shuffle=False, seed=2021)
null_imp_df = pd.DataFrame()
nb_runs = 20
for i in tqdm(range(nb_runs)):
    imp_df = get_feature_importances(data=pd.concat([X_train, X_train_], axis=0), shuffle=True)
    imp_df['run'] = i + 1
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    correlation_scores.append((_f, split_score, gain_score))
corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
features = [_f for _f, _, _score in correlation_scores if _score >= 40]
cat_cols = [_f for _f, _, _score in correlation_scores if (_score >= 40) & (_f in cat_cols)]

print("得到最终特征共计{}维".format(len(features)))

print("开始模型二的训练与预测")
# 模型训练、预测
KF = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
X_train[cat_cols] = X_train[cat_cols].astype('category')
X_test[cat_cols] = X_test[cat_cols].astype('category')
params = {
          'objective':'binary',
          'metric':'binary_error',
          'learning_rate':0.05,
          'subsample':0.8,
          'subsample_freq':3,
          'colsample_btree':0.8,
          'num_iterations': 10000,
          'silent':True
}
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros((len(X_test)))

# 五折交叉验证
for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train.values, y.values)):
    print("fold n°{}".format(fold_))
    print('trn_idx:',trn_idx)
    print('val_idx:',val_idx)
    t = pd.concat([X_train.iloc[trn_idx][features], X_train_.iloc[trn_idx][features]], axis=0)
    t[cat_cols] = t[cat_cols].astype('category')
    trn_data = lgb.Dataset(t,label=list(y.iloc[trn_idx]) + list(y_.iloc[trn_idx]) )
    t = X_train.iloc[val_idx][features]
    # t['t'] = np.nan
    val_data = lgb.Dataset(t,label=y.iloc[val_idx])
    num_round = 10000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets = [trn_data, val_data],
        verbose_eval=500,
        early_stopping_rounds=200,
        categorical_feature=cat_cols,
    )
    #feat_imp_df['imp'] += clf.feature_importance() / 5
    oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(X_test[features], num_iteration=clf.best_iteration)
print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
test['predict'] = predictions_lgb / 5
test.to_csv('lgb_predict2.csv', index=False)
print("完成模型二的训练与预测")
print("*******************************")



lgb1 = pd.read_csv('lgb_predict.csv')
lgb2 = pd.read_csv('lgb_predict2.csv')
result = []
for i in range(len(lgb1)):
    result.append((lgb1['predict'].iloc[i] + lgb2['predict'].iloc[i]) / 2)
sum([1 if i >= 0.5 else 0 for i in result])

# 后处理
post_pre = np.array(result).argsort()[::-1]
test.reset_index(drop=True, inplace=True)
test['predict'] = 0
for i in post_pre[:int(sum(y))]:
    test['predict'][i] = 1
test.to_csv('predict_2.csv', header=None, index=False)

print("完成两个模型的融合，生成提交文件")