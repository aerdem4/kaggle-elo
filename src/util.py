import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


SOME_REF_DATE = pd.to_datetime("2019-01-01")

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def plot_importance(feature_importance_df):
    cols = (feature_importance_df[["Feature", "importance"]]
            .groupby("Feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:10].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(14,26))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')

    
def sanity_check(train_target, test_target):
    print("Test target min:", test_target.min())
    assert (test_target < train_target.min()).sum() == 0
    print("Test target max:", test_target.max())
    assert (test_target > train_target.max()).sum() == 0
    print("Train-test mean target diff:", train_target.mean() - test_target.mean())
    assert np.abs(train_target.mean() - test_target.mean()) < 0.01
    
    test_target.hist(bins=100, alpha=0.5, normed=True)
    train_target.hist(bins=100, alpha=0.5, normed=True)
    
    
def create_new_columns(name, aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


def preprocess_hist(df):
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    df['category_3'] = df['category_3'].map({'A':0, 'B':1, 'C':2}) 
    df['month_diff'] = ((SOME_REF_DATE - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']

    
def aggregate_transactions(df, prefix):
    aggs = {}

    for col in ['subsector_id','merchant_id','merchant_category_id']:
        aggs[col] = ['nunique']
    for col in ['month', 'hour', 'weekofyear', 'dayofweek', 'year']:
        aggs[col] = ['nunique', 'mean', 'min', 'max']

    aggs['purchase_amount'] = ['sum','max','min','mean','var']
    aggs['installments'] = ['sum','max','min','mean','var']
    aggs['purchase_date'] = ['max','min']
    aggs['month_lag'] = ['max','min','mean','var']
    aggs['month_diff'] = ['mean', 'min', 'max', 'var']
    aggs['authorized_flag'] = ['sum', 'mean', 'min', 'max']
    aggs['weekend'] = ['sum', 'mean', 'min', 'max']
    aggs['category_1'] = ['sum', 'mean', 'min']
    aggs['category_2'] = ['sum', 'mean', 'min']
    aggs['category_3'] = ['sum', 'mean', 'min']
    aggs['card_id'] = ['size', 'count']

    for col in ['category_2','category_3']:
        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')
        aggs[col+'_mean'] = ['mean']    

    new_columns = create_new_columns(prefix, aggs)
    group_df = df.groupby('card_id').agg(aggs)
    group_df.columns = new_columns
    group_df.reset_index(drop=False,inplace=True)
    group_df[prefix + '_purchase_date_diff'] = (group_df[prefix + '_purchase_date_max'] - group_df[prefix + '_purchase_date_min']).dt.days
    group_df[prefix + '_purchase_date_average'] = group_df[prefix + '_purchase_date_diff']/group_df[prefix + '_card_id_size']
    group_df[prefix + '_purchase_date_uptonow'] = (SOME_REF_DATE - group_df[prefix + '_purchase_date_max']).dt.days
    group_df[prefix + '_purchase_date_uptomin'] = (SOME_REF_DATE - group_df[prefix + '_purchase_date_min']).dt.days
    return group_df


def extract_features(df):
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (SOME_REF_DATE - df['first_active_month']).dt.days
    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days
    df['hist_new_first_buy'] = (df['hist_new_purchase_date_min'] - df['first_active_month']).dt.days
    df['hist_new_last_buy'] = (df['hist_new_purchase_date_max'] - df['first_active_month']).dt.days
    
    df["time_to_first_new"] = (df['hist_new_purchase_date_min'] - df['hist_purchase_date_max']).dt.days
    df["time_to_last_new"] = (df['hist_new_purchase_date_max'] - df['hist_purchase_date_max']).dt.days
    
    for f in ['hist_purchase_date_max','hist_purchase_date_min','hist_new_purchase_date_max', 'hist_new_purchase_date_min']:
        df[f] = df[f].astype(np.int64) * 1e-9
        
    df['card_id_total'] = df['hist_new_card_id_size']+df['hist_card_id_size']
    df['card_id_cnt_total'] = df['hist_new_card_id_count']+df['hist_card_id_count']
    df['purchase_amount_total'] = df['hist_new_purchase_amount_sum']+df['hist_purchase_amount_sum']
    df['purchase_amount_mean'] = df['hist_new_purchase_amount_mean']+df['hist_purchase_amount_mean']
    df['purchase_amount_max'] = df['hist_new_purchase_amount_max']+df['hist_purchase_amount_max']
