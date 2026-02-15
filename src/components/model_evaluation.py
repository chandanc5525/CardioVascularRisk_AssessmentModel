import os
from sklearn.metrics import r2_score, mean_absolute_error
from src.logger import get_logger
from src.exceptions import CustomException
from src.utils import save_object, save_metrics
from src.config import MODEL_PATH, METRICS_PATH

logger = get_logger(__name__)


class ModelEvaluation:

    def evaluate_models(self, models, X_test, y_test):
        try:
            logger.info("Starting model evaluation")

            best_score = -1
            best_model = None
            best_model_name = None

            metrics_dict = {}

            for name, model in models.items():
                preds = model.predict(X_test)

                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)

                metrics_dict[name] = {
                    "r2_score": r2,
                    "mae": mae
                }

                logger.info(f"{name} -> R2: {r2}, MAE: {mae}")

                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name

            # Save best model
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            save_object(MODEL_PATH, best_model)

            # Save metrics
            save_metrics(METRICS_PATH, metrics_dict)

            logger.info(f"Best Model: {best_model_name}")
            logger.info("Best model and metrics saved successfully")

            return best_model_name, best_score

        except Exception as e:
            logger.error("Error in model evaluation")
            raise CustomException(str(e))
