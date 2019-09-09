"""
Import and play for tree-based models.
"""


def lgbm_model_oof(train_X, train_y, test_X, n_folds, lgbm_params, random_state=7557):

    folds = KFold(n_splits=n_folds, shuffle=False, random_state=random_state)
    oof = np.zeros(len(train_X))
    predictions = np.zeros(len(test_X))
    feature_importance_df = pd.DataFrame()
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X, train_y)):
        print("Fold {}".format(fold_))
        
        trn_data = lgb.Dataset(
            train_X.iloc[trn_idx][features], label=train_y.iloc[trn_idx], free_raw_data=True)
        val_data = lgb.Dataset(
            train_X.iloc[val_idx][features], label=train_y.iloc[val_idx], free_raw_data=True)

        num_round = 1000
        clf = lgb.train(lgbm_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000,
                        early_stopping_rounds = 200)
        oof[val_idx] = clf.predict(train_X.iloc[val_idx][features], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test_X, num_iteration=clf.best_iteration) / folds.n_splits
        
        del trn_data, val_data
        gc.collect()

    return predictions, oof, feature_importance_df