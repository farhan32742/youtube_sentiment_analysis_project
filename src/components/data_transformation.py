import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.logger.logging import logging
from src.exception.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from src.utils.utils import save_object
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Download NLTK resources (do this once)
nltk.download('stopwords')
nltk.download('wordnet')

@dataclass
class TextDataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'text_preprocessor.pkl')

class TextDataTransformation:
    def __init__(self):
        self.text_data_transformation_config = TextDataTransformationConfig()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}

    def text_preprocessor(self, text):
        """Custom text preprocessing function"""
        try:
            # Lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            
            # Tokenize and lemmatize
            tokens = text.split()
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            raise CustomException(e, sys)

    def get_transformation_pipeline(self):
        try:
            # Text processing pipeline
            text_pipeline = Pipeline(
                
                steps=[
                    ('tfidf', TfidfVectorizer(
                        preprocessor=self.text_preprocessor,
                        max_features=5000,  # Limit number of features
                        ngram_range=(1, 2)  # Consider unigrams and bigrams
                    ))
                ]
            )
            
            # Alternative: Could also add CountVectorizer in a ColumnTransformer
            # if you have multiple text columns
            
            return text_pipeline

        except Exception as e:
            logging.info("Error in creating text transformation pipeline")
            raise CustomException(e, sys)

    def initiate_text_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test text data completed")
            
            # Assuming your text column is named 'text'
            text_column = 'clean_comment' 
            print(train_df[text_column].head())
            train_df = train_df.dropna(subset=[text_column])
            test_df = test_df.dropna(subset=[text_column])
            train_df = train_df.drop_duplicates(subset=[text_column])
            test_df = test_df.drop_duplicates(subset=[text_column])
            train_df = train_df[~(train_df[text_column].str.strip() == '')]



            preprocessing_obj = self.get_transformation_pipeline()
            
            # Fit on training data
            train_features = preprocessing_obj.fit_transform(train_df[text_column])
            test_features = preprocessing_obj.transform(test_df[text_column])
            
            # If you have a target variable
            target_column_name = 'category'  # Change to your target column
            train_target = train_df[target_column_name].values
            test_target = test_df[target_column_name].values
            
            # Save preprocessing object
            save_object(
                file_path=self.text_data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("Text preprocessing pickle file saved")
            
            return (
                train_features,
                test_features,
                train_target,
                test_target
            )
            
        except Exception as e:
            logging.info("Error in text data transformation")
            raise CustomException(e, sys)