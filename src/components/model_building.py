from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.logger import get_logger
from src.exceptions import CustomException

logger = get_logger(__name__)


class ModelBuilding:

    def __init__(self):
        
        self.models = {
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "AdaBoostRegressor": AdaBoostRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "SVR": SVR(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42)
        }

    def train_models(self, X_train, y_train):
        
        try:
            logger.info("Starting model training")

            trained_models = {}

            for name, model in self.models.items():
                logger.info(f"Training {name}")
                model.fit(X_train, y_train)
                trained_models[name] = model

            logger.info("All models trained successfully")

            return trained_models

        except Exception as e:
            logger.error("Error in model building")
            raise CustomException(str(e))
