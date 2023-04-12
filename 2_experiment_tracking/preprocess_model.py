import pandas as pd

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR

from sklearn.metrics import mean_squared_error

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment-1")

class ModelTracking:
    
    def __init__(self) -> None:
        pass

    def read_dataframe(self, filename:str):
        self.df = pd.read_parquet(filename)

        self.df.lpep_dropoff_datetime = pd.to_datetime(self.df.lpep_dropoff_datetime)
        self.df.lpep_pickup_datetime = pd.to_datetime(self.df.lpep_pickup_datetime)

        self.df['duration'] = self.df.lpep_dropoff_datetime - self.df.lpep_pickup_datetime
        self.df.duration = self.df.duration.apply(lambda td: td.total_seconds() / 60)

        self.df = self.df[(self.df.duration >= 1) & (self.df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        self.df[categorical] = self.df[categorical].astype(str)

        self.df['PU_DO'] = self.df['PULocationID'] + '_' + self.df['DOLocationID']
        
        return self.df

    def feature_target_set(self, df_train, df_val):
        self.categorical = ["PU_DO"]
        self.numerical = ["trip_distance"]
        self.target = ["trip_distance"]

        self.dv = DictVectorizer()
        
        self.train_dict = df_train[self.categorical + self.numerical].to_dict(orient="records")
        self.X_train = self.dv.fit_transform(self.train_dict)

        self.val_dict = df_val[self.categorical + self.numerical].to_dict(orient="records")
        self.X_val = self.dv.transform(self.val_dict)

        self.y_train = df_train[self.target].values
        self.y_val = df_val[self.target].values

        with open("./models/dv_nyc_taxi.bin", "wb") as self.f_out:
            pickle.dump(self.dv, self.f_out)

        return self.X_train, self.X_val, self.y_train, self.y_val

    
    def lasso_tracker(self, 
                    ds_name:str, 
                    train_path:str, 
                    val_path:str, 
                    alpha:float,
                    model_name:str, 
                    X_train, y_train,
                    X_val, y_val):
        self.alpha = alpha
        with mlflow.start_run():
            mlflow.set_tag("data_scientist", ds_name)
            mlflow.log_param("train-data-path", train_path)
            mlflow.log_param("val-data-path", val_path)

            mlflow.log_param("alpha", self.alpha)

            self.model = Lasso(alpha=self.alpha)
            self.model.fit(X_train, y_train)

            self.y_pred = self.model.predict(X_val)
            self.rmse = mean_squared_error(y_val, self.y_pred, squared=False)
            mlflow.log_metric("rmse", self.rmse)

            with open(f"./models/{model_name}", "wb") as f_out:
                pass

            mlflow.log_artifact(local_path=f"./models/{model_name}", 
                                artifact_path="models_pickle")

        return "Done"     
    
    def multi_model_tracking(self,
                             X_train, y_train, 
                             X_val, y_val,
                             train_path:str, val_path:str,
                             ds_name:str):
        mlflow.sklearn.autolog()
        for self.model_class in (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, LinearSVR):
            with mlflow.start_run():
                mlflow.set_tag("data_scientist", ds_name)
                mlflow.log_param("train-data-path", train_path)
                mlflow.log_param("val-data-path", val_path)

                self.model = self.model_class()
                self.model.fit(X_train, y_train)

                self.y_pred = self.model.predict(X_val)
                self.rmse = mean_squared_error(y_val, self.y_pred, squared=False)
                mlflow.log_metric("rmse", self.rmse)
                mlflow.sklearn.log_model(self.model, artifact_path="models_nyc_dur_pred")
                print(f"Model {self.model_class} done")
        return {"rmse": self.rmse}


 


if __name__ == "__main__":
    
    mt = ModelTracking()
    
    df_train = mt.read_dataframe("./dataset/green_tripdata_2022-10.parquet")
    df_val = mt.read_dataframe("./dataset/green_tripdata_2022-11.parquet")
    
    X_train, X_val, y_train, y_val = mt.feature_target_set(df_train, df_val)

    print(df_val.head())
    print(y_val)

    mt.lasso_tracker(ds_name="Soumyadip Majumder",
                    train_path="./dataset/green_tripdata_2022-10.parquet",
                    val_path="./dataset/green_tripdata_2022-11.parquet",
                    alpha=0.1,
                    model_name="lasso_nyc_pred_1.bin",
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val)

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    def xgb_objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=30,
                evals = [(valid, "validation")],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=xgb_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials())

    print(best_result)

    with mlflow.start_run():

        best_params = {'learning_rate': 0.535888083114419, 
                       'max_depth': 60, 
                       'min_child_weight': 1.086162666734462, 
                       'reg_alpha': 0.06182351110027792, 
                       'reg_lambda': 0.00257279377193601}

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

