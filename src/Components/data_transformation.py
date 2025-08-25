import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# These imports are assumed to be from your project structure
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def add_time_features(self, df):
        """Extract raw + cyclical + new time features from Timestamp"""
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Hour"] = df["Timestamp"].dt.hour
        df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
        
        # A simple heuristic for peak hours (e.g., morning and evening)
        df["Peak_Hours"] = df["Hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Check if the day of the week is a weekend (Saturday=5, Sunday=6)
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
        
        # Original cyclical features
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
        df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
        
        return df

    def get_data_transformer_object(self):
        try:
            # Numerical features
            numerical_columns = [
                "Temperature(°C)", "Occupancy", "Appliance_Usage(kWh)",
                "Smart_Lighting(kWh)", "Thermostat_Setting(°C)",
                "Hour", "DayOfWeek",
                "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
                "Peak_Hours", "IsWeekend"
            ]

            # True categorical
            categorical_columns = ["Weather", "Alert"]

            # Pipelines
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Removed the StandardScaler from the categorical pipeline
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data read successfully")

            # Add engineered features
            train_df = self.add_time_features(train_df)
            test_df = self.add_time_features(test_df)

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Energy_Consumption(kWh)"

            X_train = train_df.drop(columns=[target_column_name, "Timestamp"], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name, "Timestamp"], axis=1)
            y_test = test_df[target_column_name]

            logging.info("Applying preprocessing object")

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_transformed, np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys)
