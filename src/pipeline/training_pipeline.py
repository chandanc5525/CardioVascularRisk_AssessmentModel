from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_building import ModelBuilding
from src.components.model_evaluation import ModelEvaluation


class TrainingPipeline:

    def run_pipeline(self):

        # 1. Ingestion
        ingestion = DataIngestion()
        train_df, test_df = ingestion.data_ingestion()

        # 2. Preprocessing
        preprocessing = DataPreprocessing()
        X_train, X_test, y_train, y_test = preprocessing.preprocess(train_df, test_df)

        # 3. Model Training
        model_builder = ModelBuilding()
        trained_models = model_builder.train_models(X_train, y_train)

        # 4. Model Evaluation
        evaluator = ModelEvaluation()
        best_model_name, best_score = evaluator.evaluate_models(
            trained_models,
            X_test,
            y_test
        )

        return {
            "best_model": best_model_name,
            "best_r2_score": best_score
        }
