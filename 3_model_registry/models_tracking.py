import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

os.environ["AWS_PROFILE"] = "soumyadip"

TRACKING_SERVER_HOST = "13.232.106.183"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

MLFLOW_TRACKING_URI = "postgresql://mlflow:crookednoob@mlflow-database.cys5yhae99jt.ap-south-1.rds.amazonaws.com:5432/mlflow_db" #"sqlite:///mlflow.db"
client = MlflowClient()#MLFLOW_TRACKING_URI)

#mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("nyc-green-taxi-models-2")

class ModelRegistry:
    
    def __init__(self) -> None:

        self.experiment_id = "5"


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

        with open("./models/dv_preprocessor.bin", "wb") as self.f_out:
            pickle.dump(self.dv, self.f_out)

        return self.X_train, self.X_val, self.y_train, self.y_val
    

    def multi_model_training(self, 
                             X_train, y_train,
                             X_val, y_val,
                             train_path:str, val_path:str,
                             ds_name:str):
        mlflow.sklearn.autolog()
        for self.model_class in (LinearRegression, Lasso, 
                                 Ridge, GradientBoostingRegressor, 
                                 ExtraTreesRegressor, RandomForestRegressor, 
                                 SVR):
            with mlflow.start_run():
                mlflow.set_tag("data_scientist", ds_name)
                mlflow.log_param("train_path", train_path)
                mlflow.log_param("val_path", val_path)

                self.model = self.model_class()
                self.model.fit(X_train, y_train)

                self.y_pred = self.model.predict(X_val)

                self.rmse = mean_squared_error(y_val, self.y_pred)
                mlflow.log_metric("rmse", self.rmse)
                mlflow.sklearn.log_model(self.model, artifact_path="prediction_model")
                mlflow.log_artifact("models/dv_preprocessor.bin", artifact_path="preprocessor_model")
                print(f"\n\nModel {self.model_class.__name__} training complete\n\n")
        return "Completed"

    def register_model(self):

#        print(client.search_runs(experiment_ids="5"))
        runs = client.search_runs(
                            experiment_ids=self.experiment_id,
                            filter_string="metrics.rmse < 1",
                            run_view_type=ViewType.ACTIVE_ONLY,
                            max_results=3,
                            order_by=["metrics.rmse ASC"])
        for run in runs:
            print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']}:.4f")




if __name__ == "__main__":
    
    mr = ModelRegistry()

    train_path = "./dataset/green_tripdata_2022-10.parquet"
    val_path = "./dataset/green_tripdata_2022-11.parquet"

    df_train = mr.read_dataframe(train_path)
    df_val = mr.read_dataframe(val_path)
    
    X_train, X_val, y_train, y_val = mr.feature_target_set(df_train, df_val)

    mr.multi_model_training(ds_name="Soumyadip Majumder",
                            train_path=train_path,
                            val_path=val_path,
                            X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val)

    mr.register_model()