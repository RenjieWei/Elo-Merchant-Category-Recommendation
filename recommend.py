import numpy as np 
import pandas as pd 
import os
import lightgbm as lgb
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
import datetime
import time
import sys

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# loading the dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
histdata = pd.read_csv("../input/historical_transactions.csv")
newdata = pd.read_csv("../input/new_merchant_transactions.csv")
merchants = pd.read_csv("../input/merchants.csv")

# for submission
submission = pd.DataFrame({"card_id":test["card_id"].values})

# Feature Engnineering
target = train["target"]

# Covert time as features
for data in [train,test]:
    data['first_active_month'] = pd.to_datetime(data['first_active_month'])
    data['year'] = data['first_active_month'].dt.year
    data['month'] = data['first_active_month'].dt.month
    data['elapsed_time'] = (datetime.date(2018,2,1) - data['first_active_month'].dt.date).dt.days

train = train.drop(['first_active_month','target'], axis=1)
test = test.drop(['first_active_month'], axis=1)

# Convert category values
def category_convert(data):
    data['cat2'] = data['category_2']
    data['cat3'] = data['category_3']
    data = pd.get_dummies(data, columns=['cat2', 'cat3'])
    for bi_cat in ['authorized_flag', 'category_1']:
        data[bi_cat] = data[bi_cat].map({'Y':1, 'N':0})
    return data

histdata = category_convert(histdata)
newdata = category_convert(newdata)

# historical_transactions & new merchants transaction
# categorical data and other general data
def aggregate_trans(data, prefix):  
    agg_func = {
        'card_id': ['size'], #num_trans
        'authorized_flag': ['sum', 'mean','nunique'],
        'category_1': ['sum', 'mean','nunique'],
        'category_2': ['nunique'],
        'category_3': ['nunique'],
        'cat2_1.0': ['mean'],
        'cat2_2.0': ['mean'],
        'cat2_3.0': ['mean'],
        'cat2_4.0': ['mean'],
        'cat2_5.0': ['mean'],
        'cat3_A': ['mean'],
        'cat3_B': ['mean'],
        'cat3_C': ['mean'],
        'city_id': ['nunique'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique'],
        'installments': ['sum', 'mean','median', 'max', 'min', 'std', 'nunique'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'month_lag': ['mean', 'max', 'min', 'std', 'nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std', 'nunique']
    }    
    agg_trans = data.groupby(['card_id']).agg(agg_func)
    agg_trans.columns = [prefix + '_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)
    
    return agg_trans

hist_sum = aggregate_trans(histdata, 'hist_')
new_sum = aggregate_trans(newdata, 'new_')

# Devide time 
# historical / new transaction
def devide_time(data):
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    data['month_diff'] = ((datetime.datetime.today() - data['purchase_date']).dt.days)//30  
    data['purchase_year'] = data['purchase_date'].dt.year
    data['purchase_month'] = data['purchase_date'].dt.month
    data['weekofyear'] = data['purchase_date'].dt.weekofyear
    data['dayofweek'] = data['purchase_date'].dt.dayofweek
    data['weekend'] = (data.purchase_date.dt.weekday >=5).astype(int)
    data['hour'] = data['purchase_date'].dt.hour
    return data

hist_times = devide_time(histdata)
new_times = devide_time(newdata)

def aggregate_times(data, prefix):  

    agg_func = {

        'month_diff': ['mean','max','min'],
        'purchase_year': ['mean', 'max', 'min', 'std','nunique'],
        'purchase_month': ['mean', 'max', 'min', 'std','nunique'],
        'weekofyear': ['mean','max','min','nunique'],
        'dayofweek': ['mean'],
        'weekend': ['sum', 'mean'],
        'hour': ['mean','max','min']
    }    
    agg_times = data.groupby(['card_id']).agg(agg_func)
    agg_times.columns = [prefix + '_'.join(col).strip() 
                           for col in agg_times.columns.values]
    agg_times.reset_index(inplace=True)
    
    return agg_times

hist_times = aggregate_times(hist_times, 'hist_')
new_times = aggregate_times(new_times, 'new_')

# purchase date term
histdata['pur_date'] = pd.DatetimeIndex(histdata['purchase_date']).date
newdata['pur_date'] = pd.DatetimeIndex(newdata['purchase_date']).date

histdata.loc[:,'pur_date'] = pd.DatetimeIndex(histdata['pur_date']).astype(np.int64) * 1e-9
newdata.loc[:,'pur_date'] = pd.DatetimeIndex(newdata['pur_date']).astype(np.int64) * 1e-9

agg_fn= {
        'pur_date': [np.ptp,'max','min'], # np.ptp: Range of values (maximum - minimum) 
        }
agg_hist = histdata.groupby(['card_id']).agg(agg_fn)
agg_hist.columns = ['_'.join(col).strip() for col in agg_hist.columns.values]
agg_hist.reset_index(inplace=True)

agg_new = newdata.groupby(['card_id']).agg(agg_fn)
agg_new.columns = ['_'.join(col).strip() for col in agg_new.columns.values]
agg_new.reset_index(inplace=True)

agg_hist.columns = ['hist_' + c if c != 'card_id' else c for c in agg_hist.columns]
agg_new.columns = ['new_' + c if c != 'card_id' else c for c in agg_new.columns]

# scale agg_hist, agg_new

agg_hist['hist_pur_date_ptp']=sk.preprocessing.scale(agg_hist['hist_pur_date_ptp'])
agg_new['new_pur_date_ptp']=sk.preprocessing.scale(agg_new['new_pur_date_ptp'])
agg_hist['hist_pur_date_max']=sk.preprocessing.scale(agg_hist['hist_pur_date_max'])
agg_new['new_pur_date_max']=sk.preprocessing.scale(agg_new['new_pur_date_max'])
agg_hist['hist_pur_date_min']=sk.preprocessing.scale(agg_hist['hist_pur_date_min'])
agg_new['new_pur_date_min']=sk.preprocessing.scale(agg_new['new_pur_date_min'])

# merge 
hist = hist_times.merge(hist_sum,on='card_id',how='left')
hist = hist.merge(agg_hist, on='card_id',how='left')
del hist_sum
del hist_times
del agg_hist

new = new_times.merge(new_sum, on='card_id',how='left')
new = new.merge(agg_new, on='card_id',how='left')

del new_sum
del new_times
del agg_new


train = train.merge(hist, on='card_id',how='left')
train = train.merge(new, on='card_id',how='left')

test = test.merge(hist, on='card_id',how='left')
test = test.merge(new, on='card_id',how='left')

train.head()

# save featured data
train.to_csv("train_featured.csv", index=False)
test.to_csv("test_featured.csv", index=False)

# drop card_id before running model
train = train.drop('card_id', axis=1) #,'hist_avg_term','new_avg_term'
test = test.drop('card_id', axis=1) #,'hist_avg_term','new_avg_term'


# Light GBM
# set default parameters
params = {'boosting': 'gbdt',
          'num_trees': 1000,
          'objective':'regression',
          'metric': 'rmse',
          'learning_rate': 0.01, # 0.003! #0.005 #0.006 
          'num_leaves': 111, #110 #100 #150 large, but over-fitting
          'max_bin': 66,  #60 #50 # large,but slower,over-fitting
          'max_depth': 10, # deal with over-fitting
          'min_data_in_leaf': 149, # deal with over-fitting
          'min_child_samples': 20,
          'feature_fraction': 0.5,#0.5 #0.6 #0.8
          'bagging_fraction': 0.8,
          'bagging_freq': 1,#5  
          'bagging_seed': 11,
          'lambda_l1': 0.2634,#1.3! #5 #1.2 #1
          'lambda_l2': 0.1 ,#0.1,
          'random_state': 133,
          "verbosity": -1
         }


# Kfold cross-validation
nfolds = 5
nrepeats = 2 
folds = RepeatedKFold(n_splits=nfolds, n_repeats=nrepeats, random_state=11)
fold_pred = np.zeros(len(train))
feature_importance_df = pd.DataFrame()
lgb_preds = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values,target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx]) #categorical_feature=categorical_feats
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx]) #categorical_feature=categorical_feats

    iteration = 2000
    lgb_m = lgb.train(params, trn_data, iteration, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    fold_pred[val_idx] = lgb_m.predict(train.iloc[val_idx], num_iteration=lgb_m.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = train.columns
    fold_importance_df["importance"] = lgb_m.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    lgb_preds += lgb_m.predict(test, num_iteration=lgb_m.best_iteration) / (nfolds*nrepeats)

print("CV score: {:<8.5f}".format(np.sqrt(mean_squared_error(fold_pred, target))))
