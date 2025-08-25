import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object # Assuming load_object is in src/utils.py
import numpy as np


class PredictPipeline:
    def __init__(self):
        # File paths for your trained model and preprocessor
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        """
        Loads the preprocessor and model, transforms the input features,
        and returns the prediction.

        Args:
            features (pd.DataFrame): DataFrame containing the input features.

        Returns:
            np.array: The predicted energy consumption.
        """
        try:
            # Check if artifact files exist
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at: {self.preprocessor_path}")

            print("Before Loading")
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Temperature: float,
        Occupancy: int,
        Appliance_Usage: float,
        Smart_Lighting: float,
        Thermostat_Setting: float,
        Weather: str,
        Timestamp: str
    ):
        self.Temperature = Temperature
        self.Occupancy = Occupancy
        self.Appliance_Usage = Appliance_Usage
        self.Smart_Lighting = Smart_Lighting
        self.Thermostat_Setting = Thermostat_Setting
        self.Weather = Weather
        self.Timestamp = Timestamp

    def get_data_as_data_frame(self):
        """
        Converts the custom data input into a pandas DataFrame,
        including all necessary feature engineering.
        """
        try:
            custom_data_input_dict = {
                "Temperature(°C)": [self.Temperature],
                "Occupancy": [self.Occupancy],
                "Appliance_Usage(kWh)": [self.Appliance_Usage],
                "Smart_Lighting(kWh)": [self.Smart_Lighting],
                "Thermostat_Setting(°C)": [self.Thermostat_Setting],
                "Weather": [self.Weather],
                "Timestamp": [self.Timestamp]
            }

            df = pd.DataFrame(custom_data_input_dict)

            # Convert Timestamp to datetime and extract time-based features
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["Hour"] = df["Timestamp"].dt.hour
            df["DayOfWeek"] = df["Timestamp"].dt.dayofweek

            # Engineer Peak_Hours and IsWeekend features
            df["Peak_Hours"] = df["Hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
            df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

            # Engineer cyclical features
            df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
            df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
            df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
            df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

            # Drop the original Timestamp column before returning
            df = df.drop(columns=["Timestamp"], axis=1)

            return df

        except Exception as e:
            raise CustomException(e, sys)