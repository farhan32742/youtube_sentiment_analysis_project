
import pandas as pd
import numpy as np
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import TextDataTransformation
from src.components.model_trainer import ModelTrainer




obj = DataIngestion()

train_data_path,test_data_path=obj.initiate_Data_Ingestion()


print(train_data_path)
print(test_data_path)
data_transformation=TextDataTransformation()
train_features, test_features, train_target, test_target = data_transformation.initiate_text_data_transformation(train_data_path, test_data_path)

model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_trainer(train_features, test_features, train_target, test_target)