import pandas as pd
from preprocess_model import ModelTracking

mt = ModelTracking()

df_train = mt.read_dataframe("./dataset/green_tripdata_2022-10.parquet")
df_val = mt.read_dataframe("./dataset/green_tripdata_2022-11.parquet")

X_train, X_val, y_train, y_val = mt.feature_target_set(df_train, df_val)


# mt.lasso_tracker(ds_name="Soumyadip Majumder",
#                 train_path="./dataset/green_tripdata_2022-10.parquet",
#                 val_path="./dataset/green_tripdata_2022-11.parquet",
#                 alpha=0.05,
#                 model_name="lasso_nyc_pred_1.bin",
#                 X_train=X_train, y_train=y_train,
#                 X_val=X_val, y_val=y_val)

mt.multi_model_tracking(ds_name="Soumyadip Majumder",
            train_path="./dataset/green_tripdata_2022-10.parquet",
            val_path="./dataset/green_tripdata_2022-11.parquet",
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val)