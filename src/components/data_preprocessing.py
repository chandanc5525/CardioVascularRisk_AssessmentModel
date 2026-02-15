import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.logger import get_logger
from src.exceptions import CustomException
from src.utils import save_object
from src.config import PROCESSED_DIR

logger = get_logger(__name__)

TARGET_COLUMN = "heart_disease_risk_score"


class DataPreprocessing:

    def preprocess(self, train_df, test_df):
        try:
            logger.info("Starting data preprocessing")

            # Remove duplicates
            train_df = train_df.drop_duplicates()
            test_df = test_df.drop_duplicates()

            # Drop ID column
            train_df = train_df.drop(columns=["Patient_ID"])
            test_df = test_df.drop(columns=["Patient_ID"])

            # Split features and target
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            # Identify feature types
            categorical_features = X_train.select_dtypes(include=["object"]).columns
            numerical_features = X_train.select_dtypes(exclude=["object"]).columns

            logger.info(f"Categorical columns: {list(categorical_features)}")
            logger.info(f"Numerical columns: {list(numerical_features)}")

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", MinMaxScaler(), numerical_features),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
                ]
            )

            # Fit only on train
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Save preprocessor
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            save_object(os.path.join(PROCESSED_DIR, "preprocessor.pkl"), preprocessor)

            logger.info("Preprocessing completed successfully")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error("Error in data preprocessing")
            raise CustomException(str(e))
