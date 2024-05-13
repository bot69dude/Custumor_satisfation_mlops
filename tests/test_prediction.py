import os
import sys
# Add the path to your project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from src.mlProject.components.prediction import PredictionPipeline

@pytest.fixture
def prediction_model():
    return PredictionPipeline()

def test_prediction(prediction_model):
    input_data = ['delivered','credit_card',3,18.59,'florianopolis','MG',339.0,7.78,664.0,6,'cool_stuff',6,4,5,-2]
    prediction = prediction_model.predict(input_data)
    # Add assertions to validate the prediction
    assert prediction >= 0 and prediction <= 5, "prediction should be between 0 and 5"