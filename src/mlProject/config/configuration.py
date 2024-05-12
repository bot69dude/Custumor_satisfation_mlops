from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml, create_directories
from src.mlProject.entity.config_entity import (DataIngestionConfig,
                                                DataPreprocessConfig,
                                                 DataTransformationConfig,
                                                 ModelTrainerConfig,
                                                 ModelEvaluationConfig)
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE,
        params_filepath = PARAMS_FILE):

        config_filepath = Path(config_filepath)
        params_filepath = Path(params_filepath)

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessConfig:
        config = self.config.data_preprocessing
        drop_columns = self.params['features_to_drop']
        cat_columns = self.params['categorical_features']
        
        create_directories([config.root_dir])
        data_preprocessing_config = DataPreprocessConfig(
            root_dir = config.root_dir,
            data_path = config.data_dir,
            cat_params = cat_columns,
            drop_params = drop_columns  # Pass DROP_COLUMNS to DataPreprocessConfig
        )
        
        return data_preprocessing_config 
    
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.XGBRegressor
        schema =  self.params.TARGET_COLUMN
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            objective = params.objective,
            n_estimators = params.n_estimators,
            learning_rate = params.learning_rate, 
            max_depth = params.max_depth,
            min_child_weight = params.min_child_weight,
            subsample = params.subsample,
            colsample_bytree = params.colsample_bytree,
            reg_lambda = params.reg_lambda,
            target_column = schema.name
            
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.XGBRegressor
        schema =  self.params.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri="https://dagshub.com/bot69dude/Custumor_satisfation_mlops.mlflow",
           
        )

        return model_evaluation_config
