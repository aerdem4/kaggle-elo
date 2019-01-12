import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


def train_and_predict(train_df, test_df, target, param, features, categorical_features, num_folds):
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    oof_preds = np.zeros(len(train_df))

    test_preds = np.zeros(len(test_df))

    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['outliers'].values)):
        print("fold {}".format(fold_))
        
        dev_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        if target == "target":
            #remove outliers
            dev_df = dev_df[dev_df["outliers"] == 0]
            val_df = val_df[val_df["outliers"] == 0]
        
        
        trn_data = lgb.Dataset(dev_df[features], dev_df[target], categorical_feature=categorical_features)
        val_data = lgb.Dataset(val_df[features], val_df[target], categorical_feature=categorical_features)

        clf = lgb.train(param, trn_data, num_boost_round=10000, valid_sets = [trn_data, val_data], 
                        verbose_eval=200, early_stopping_rounds = 100)

        oof_preds[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
        test_preds += clf.predict(test_df[features], num_iteration=clf.best_iteration) / num_folds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = clf.feature_importance(importance_type="gain")
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    return oof_preds, test_preds, feature_importance_df