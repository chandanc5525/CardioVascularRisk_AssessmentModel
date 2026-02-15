import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, PROCESSED_DIR
from src.logger import get_logger
from src.exceptions import CustomException

logger = get_logger(__name__)


class DataIngestion:

    def data_ingestion(self):
        try:
            logger.info("Starting data ingestion")

            # Read dataset
            df = pd.read_csv(DATA_PATH)
            logger.info(f"Dataset loaded successfully with shape: {df.shape}")

            # Create processed directory
            os.makedirs(PROCESSED_DIR, exist_ok=True)

            # Train-test split
            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save files
            train_path = os.path.join(PROCESSED_DIR, "train.csv")
            test_path = os.path.join(PROCESSED_DIR, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info("Train-test split completed and saved")

            return train_df, test_df

        except Exception as e:
            logger.error("Error occurred during data ingestion")
            raise CustomException(str(e))
