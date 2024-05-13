import joblib
import numpy as np
import pandas as pd
from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml
from pathlib import Path



class PredictionPipeline:
    def __init__(self,params_filepath = PARAMS_FILE):
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))
        self.params = read_yaml(params_filepath)
        
    def predict(self , input_data):
        
        model_features = list(self.model.feature_names_in_)
        
        feature = ['order_status', 'payment_type', 'payment_installments',
       'payment_value', 'customer_city', 'customer_state', 'price',
       'freight_value', 'product_description_lenght',
       'product_photos_qty', 'product_category_name_english',
       'Order_purchase_hour', 'Order_purchase_dayofweek',
       'Time_taken_for_delivery', 'delivery_delay']
        
        cat_columns = self.params['categorical_features']
        
        input_data = pd.DataFrame([input_data],columns=feature)
        input_data = pd.get_dummies(input_data,columns=cat_columns)
        
        input_list = list(input_data.columns)
        
        data_dict = {}
        for column_name in input_list:
            data_dict[column_name] = input_data[column_name][0]
            
        inp = []
        for i in model_features:
            if i in data_dict:
                inp.append(data_dict[i])
            else:
                inp.append(False)
        
        input_datax = pd.DataFrame([inp],columns=model_features)
        
        prediction = self.model.predict(input_datax)
        
        return prediction[0]