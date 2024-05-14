# Predicting Customer Satisfaction Score with Olist Dataset

## Overview
This project aims to predict the customer satisfaction score for the next order or purchase based on historical data using the Brazilian E-Commerce Public Dataset provided by Olist. By leveraging machine learning techniques, we strive to build a robust predictive model that can anticipate customer satisfaction levels and enhance overall service quality.

## Problem Statement
For a given customer's historical data, we are tasked with predicting the review score for the next order or purchase. The dataset comprises information on 100,000 orders spanning from 2016 to 2018, gathered from various marketplaces in Brazil. It encompasses diverse dimensions, including order status, price, payment, freight performance, customer location, product attributes, and customer reviews.

## Dataset
The Brazilian E-Commerce Public [Dataset Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) serves as the foundation for our analysis and model development. This comprehensive dataset offers insights into the intricacies of e-commerce operations, enabling us to extract valuable patterns and trends to inform our predictive model.

## Tech Stack
- **ML Framework**: XGBoost
- **Web Framework**: Flask
- **Experimentation Tracking**: MLflow, DVC (Data Version Control)

The Brazilian E-Commerce Public Dataset by Olist serves as the foundation for our analysis and model development. This comprehensive dataset offers insights into the intricacies of e-commerce operations, enabling us to extract valuable patterns and trends to inform our predictive model.Download the Olist dataset from [here](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## Python Requirements
Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/bot69dude/Custumor_satisfation_mlops.git
pip install -r requirements.txt
```

# Set up the MLflow tracking URI using the provided credentials:
```bash
    export MLFLOW_TRACKING_URI=https://dagshub.com/bot69dude/Retail_price_optimization.mlflow
    export MLFLOW_TRACKING_USERNAME=bot69dude
    export MLFLOW_TRACKING_PASSWORD=559b04e28f7af9242d3e209229040403de58f073
```

# Run the following commands:
```bash
    dvc repro
    python app.py
```
# Application Testing:
```bash
    pytest
```
## Experimentation Tracking
Experimentation tracking for this project is available on [Dagshub](https://dagshub.com/bot69dude/Custumor_satisfation_mlops.mlflow). You can view detailed experiment logs, metrics, and visualizations to understand the model's performance.


## Acknowledgements
We would like to express our gratitude to Olist for providing the invaluable dataset for this project.

## License
This project is licensed under the [MIT License](link-to-license).
