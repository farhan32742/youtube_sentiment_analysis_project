import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger.logging import logging
from src.exception.exception import CustomException
from src.utils.utils import save_object, evaluate_model 

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'text_classifier_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_features, test_features, train_target, test_target):
        try:
            logging.info('Training classification models on text features...')

            models = {
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'MultinomialNB': MultinomialNB(),
                'RandomForestClassifier': RandomForestClassifier(),
                'SVM': SVC(probability=True)
            }

            scores = {}
            for name, model in models.items():
                model.fit(train_features, train_target)
                predictions = model.predict(test_features)
                acc = accuracy_score(test_target, predictions)
                scores[name] = acc
                logging.info(f"{name} Accuracy: {acc}")

            best_model_name = max(scores, key=scores.get)
            best_model = models[best_model_name]
            best_score = scores[best_model_name]

            print(f"✅ Best Model: {best_model_name}, Accuracy: {best_score}")
            logging.info(f"✅ Best Model: {best_model_name}, Accuracy: {best_score}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_score

        except Exception as e:
            logging.error("Error occurred in model training.")
            raise CustomException(e, sys)
