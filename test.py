from src.mlProject.components.prediction import PredictionPipeline
from src.mlProject import logger

STAGE_NAME = "Prediction "
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   prediction_model = PredictionPipeline()
   input_data = ['delivered','credit_card',3,18.59,'florianopolis','MG',339.0,7.78,664.0,6,'cool_stuff',6,4,5,-2]
   logger.info(f" the predicted rating is: {prediction_model.predict(input_data)}")
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
