# https://www.kaggle.com/code/zivanwan/home-credit-lgb-more-robust

import gc
import os
import pickle
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import wandb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="home-credit",
    config={
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "max_depth": 10,
        "learning_rate": 0.05,
        "n_estimators": 200,  # 1000:0.55
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.8,
        "verbose": -1,
        "random_state": 42,
        "reg_alpha": 0.1,
        "reg_lambda": 10,
        "extra_trees": True,
        "num_leaves": 64,
        # "device": "gpu",
    },
)

config = wandb.config


class Pipeline:

    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
        df = df.drop("date_decision", "MONTH")
        return df

    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (
                df[col].dtype == pl.String
            ):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)
        return df


class Aggregator:

    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_last + expr_mean

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        return expr_max + expr_last + expr_mean

    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        # expr_count = [pl.count(col).alias(f"count_{col}") for col in cols]
        return expr_max + expr_last  # +expr_count

    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        return expr_max + expr_last

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        # expr_min = [pl.min(col).alias(f"min_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        return expr_max + expr_last

    def get_exprs(df):
        exprs = (
            Aggregator.num_expr(df)
            + Aggregator.date_expr(df)
            + Aggregator.str_expr(df)
            + Aggregator.other_expr(df)
            + Aggregator.count_expr(df)
        )

        return exprs


class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)


def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df


def read_files(regex_path, depth=None):
    chunks = []

    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)

    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df


def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = df_base.with_columns(
        month_decision=pl.col("date_decision").dt.month(),
        weekday_decision=pl.col("date_decision").dt.weekday(),
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) == "category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.int64).min
                    and c_max < np.iinfo(np.int64).max
                ):
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print(
        "Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem)
    )

    return df


# ROOT = Path("/kaggle/input/home-credit-credit-risk-model-stability")
ROOT = Path("../data")

TRAIN_DIR = ROOT / "parquet_files" / "train"
TEST_DIR = ROOT / "parquet_files" / "test"


def main():
    data_store = {
        "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
        "depth_0": [
            read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
            read_files(TRAIN_DIR / "train_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
            read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
            read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
            read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
            read_file(TRAIN_DIR / "train_other_1.parquet", 1),
            read_file(TRAIN_DIR / "train_person_1.parquet", 1),
            read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
            read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
            read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),
        ],
    }

    df_train = feature_eng(**data_store)
    print("train data shape:\t", df_train.shape)
    del data_store
    gc.collect()  # Drop the insignificant features
    df_train = df_train.pipe(Pipeline.filter_cols)

    df_train, cat_cols = to_pandas(df_train)
    df_train = reduce_mem_usage(df_train)
    print("train data shape:\t", df_train.shape)

    nums = df_train.select_dtypes(exclude="category").columns

    # df_train=df_train[nums]
    nans_df = df_train[nums].isna()
    nans_groups = {}
    for col in nums:
        cur_group = nans_df[col].sum()
        try:
            nans_groups[cur_group].append(col)
        except KeyError:
            nans_groups[cur_group] = [col]
    del nans_df
    gc.collect()

    def reduce_group(grps):
        use = []
        for g in grps:
            mx = 0
            vx = g[0]
            for gg in g:
                n = df_train[gg].nunique()
                if n > mx:
                    mx = n
                    vx = gg
                # print(str(gg)+'-'+str(n),', ',end='')
            use.append(vx)
            # print()
        print("Use these", use)
        return use

    def group_columns_by_correlation(matrix, threshold=0.8):
        # 列間の相関を計算する
        correlation_matrix = matrix.corr()

        # 列をグループ化する
        groups = []
        remaining_cols = list(matrix.columns)
        while remaining_cols:
            col = remaining_cols.pop(0)
            group = [col]
            correlated_cols = [col]
            for c in remaining_cols:
                if correlation_matrix.loc[col, c] >= threshold:
                    group.append(c)
                    correlated_cols.append(c)
            groups.append(group)
            remaining_cols = [
                c for c in remaining_cols if c not in correlated_cols
            ]

        return groups

    uses = []
    for k, v in nans_groups.items():
        if len(v) > 1:
            Vs = nans_groups[k]
            # cross_features=list(combinations(Vs, 2))
            # make_corr(Vs)
            grps = group_columns_by_correlation(df_train[Vs], threshold=0.8)
            use = reduce_group(grps)
            uses = uses + use
            # make_corr(use)
        else:
            uses = uses + v
        print("####### NAN count =", k)
    print(uses)
    print(len(uses))
    uses = uses + list(df_train.select_dtypes(include="category").columns)
    print(len(uses))
    df_train = df_train[uses]

    data_store = {
        "df_base": read_file(TEST_DIR / "test_base.parquet"),
        "depth_0": [
            read_file(TEST_DIR / "test_static_cb_0.parquet"),
            read_files(TEST_DIR / "test_static_0_*.parquet"),
        ],
        "depth_1": [
            read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
            read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
            read_files(TEST_DIR / "test_credit_bureau_a_1_*.parquet", 1),
            read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
            read_file(TEST_DIR / "test_other_1.parquet", 1),
            read_file(TEST_DIR / "test_person_1.parquet", 1),
            read_file(TEST_DIR / "test_deposit_1.parquet", 1),
            read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
            read_files(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
        ],
    }

    df_test = feature_eng(**data_store)
    print("test data shape:\t", df_test.shape)

    df_test = df_test.select(
        [col for col in df_train.columns if col != "target"]
    )

    print("train data shape:\t", df_train.shape)
    print("test data shape:\t", df_test.shape)

    df_test, cat_cols = to_pandas(df_test, cat_cols)
    df_test = reduce_mem_usage(df_test)

    del data_store
    gc.collect()

    print("Train is duplicated:\t", df_train["case_id"].duplicated().any())
    print(
        "Train Week Range:\t",
        (df_train["WEEK_NUM"].min(), df_train["WEEK_NUM"].max()),
    )
    print()
    print("Test is duplicated:\t", df_test["case_id"].duplicated().any())
    print(
        "Test Week Range:\t",
        (df_test["WEEK_NUM"].min(), df_test["WEEK_NUM"].max()),
    )

    X = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
    y = df_train["target"]
    weeks = df_train["WEEK_NUM"]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)

    params = config

    fitted_models = []
    cv_scores = []

    for idx_train, idx_valid in cv.split(X, y, groups=weeks):
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)],
        )
        wandb.log({"best_iteration": model.best_iteration_})
        wandb.log({"best_score": model.best_score_})
        # modelをwandbへ保存
        wandb.save("model.pickle")
        with open(os.path.join(wandb.run.dir, "model.pickle"), "wb") as f:
            pickle.dump(model, f)

        fitted_models.append(model)

        y_pred_valid = model.predict_proba(X_valid)[:, 1]
        auc_score = roc_auc_score(y_valid, y_pred_valid)
        wandb.log({"auc_score": auc_score})
        cv_scores.append(auc_score)

    print("CV AUC scores: ", cv_scores)
    print("Maximum CV AUC score: ", max(cv_scores))

    model = VotingModel(fitted_models)

    X_test = df_test.drop(columns=["WEEK_NUM"])
    X_test = X_test.set_index("case_id")

    lgb_pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

    df_subm = pd.read_csv(ROOT / "sample_submission.csv")
    df_subm = df_subm.set_index("case_id")

    df_subm["score"] = lgb_pred

    df_subm.head()

    df_subm.to_csv(
        f"submission_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
        index=False,
    )

    # drop columns based on feature importance
    lgb.plot_importance(
        fitted_models[2], importance_type="split", figsize=(10, 50)
    )
    plt.show()
    features = X_train.columns

    importances = fitted_models[2].feature_importances_

    feature_importance = (
        pd.DataFrame({"importance": importances, "features": features})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    wandb.log({"feature_importance": feature_importance.to_dict()})

    drop_list = []
    for i, f in feature_importance.iterrows():
        if f["importance"] < 80:
            drop_list.append(f["features"])
    print(f"Number of features which are not important: {len(drop_list)} ")

    print(drop_list)


if __name__ == "__main__":
    main()
