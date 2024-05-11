import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.mlProject.entity.config_entity import DataPreprocessConfig
import os
import logging  

class DataPreprocessing:
    def __init__(self, config: DataPreprocessConfig):
        self.config = config
        
    def _read_data(self):
        try:
            # Read data from CSV
            self.df = pd.read_csv(self.config.data_path)
        except Exception as e:
            logging.error(f"Error occurred while reading data: {str(e)}")
            raise e
    
    def _calculate_delivery_time(self):
        try:
            self.df['order_purchase_timestamp'] = pd.to_datetime(self.df['order_purchase_timestamp'])
            self.df['order_delivered_customer_date'] = pd.to_datetime(self.df['order_delivered_customer_date'])

            # Now that both columns are in datetime format, you can perform the subtraction
            self.df['Time_taken_for_delivery'] = (self.df['order_delivered_customer_date'] - self.df['order_purchase_timestamp']).dt.days
        except Exception as e:
            logging.error(f"Error occurred while calculating delivery time: {str(e)}")
            raise e
            
    def _calculate_delivery_delay(self):
        try:
            self.df['order_estimated_delivery_date'] = pd.to_datetime(self.df['order_estimated_delivery_date'])
            self.df['delivery_delay'] = (self.df['order_delivered_customer_date'] - self.df['order_estimated_delivery_date']).dt.days
        except Exception as e:
            logging.error(f"Error occurred while calculating delivery delay: {str(e)}")
            raise e
    
    def _encode_categorical_data(self):
        try:
            self.df_encoded = pd.get_dummies(self.df, columns=self.config.cat_params)
        except Exception as e:
            logging.error(f"Error occurred while encoding categorical data: {str(e)}")
            raise e
    
    def _scale_data(self):
        try:
            scaler = MinMaxScaler()

            # Reshape the 'review_score' column to a 2D array
            review_score = self.df_encoded['delivery_delay'].values.reshape(-1, 1)

            # Fit and transform the data
            scaled_review_score = scaler.fit_transform(review_score)

            # Replace the original 'review_score' column with the scaled values
            self.df_encoded['delivery_delay'] = scaled_review_score
        except Exception as e:
            logging.error(f"Error occurred while scaling data: {str(e)}")
            raise e
            
    def _drop_columns(self):
        try:
            self.df_encoded = self.df_encoded.drop(columns=self.config.drop_params)
        except Exception as e:
            logging.error(f"Error occurred while dropping columns: {str(e)}")
            raise e
    
    def _save_preprocessed_data(self):
        try:
            output_file_path = os.path.join(self.config.root_dir, "preprocessed_data.csv")
            self.df_encoded.to_csv(output_file_path, index=False)
            logging.info(f"Preprocessed data saved to: {output_file_path}")  
        except Exception as e:
            logging.error(f"Error occurred while saving preprocessed data: {str(e)}")
            raise e
    
    def preprocess_data(self):
        try:
            self._read_data()
            self._calculate_delivery_time()
            self._calculate_delivery_delay()
            self._encode_categorical_data()
            self._scale_data()
            self._drop_columns()
            self._save_preprocessed_data()
        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {str(e)}")
            raise e
