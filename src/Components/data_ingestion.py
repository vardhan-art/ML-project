# Importing required libraries
import os
import sys
from pathlib import Path

# --- Ensure package imports work even when running this file directly ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Importing Data Transformation and Model Trainer components
from src.Components.data_transformation import DataTransformation, DataTransformationConfig
from src.Components.model_trainer import ModelTrainer, ModelTrainerConfig

# Configuration class for Data Ingestion (stores file paths for train, test, and raw datasets)
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

# Main class responsible for reading raw data and splitting into train/test sets
class DataIngestion:
    def __init__(self):
        # Initialize ingestion configuration (paths)
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Build absolute path to the CSV so it works regardless of current working directory
            data_csv_path = PROJECT_ROOT / 'Rawdata' / 'smart_home_energy_v3.csv'
            if not data_csv_path.exists():
                raise FileNotFoundError(f"Input data file not found: {data_csv_path}")
            df = pd.read_csv(data_csv_path)
            logging.info('Read the dataset as dataframe')

            # Create directories if they do not exist (for saving processed data)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset into artifacts/data.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split dataset into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training dataset to artifacts/train.csv
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save testing dataset to artifacts/test.csv
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Return paths of train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # If error occurs, raise a custom exception with traceback details
            raise CustomException(e, sys)

# Entry point of the script
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"Train data saved to: {train_data}")
    print(f"Test data saved to: {test_data}")

    # Now that imports are fixed, this section will run.
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
