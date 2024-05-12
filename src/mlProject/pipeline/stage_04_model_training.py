from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.model_trainer import ModelTrainer
from src.mlProject import logger

STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_training = ModelTrainer(config=model_trainer_config)
            model_training.train()
            logger.info(f"Model trained successfully and saved at: {model_training.train()}")
        except Exception as e:
            logger.exception(f"Error occurred during {STAGE_NAME} execution: {str(e)}")
            raise e

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e