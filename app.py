from flask import Flask, render_template, request, redirect
from src.mlProject.components.prediction import PredictionPipeline
import pandas as pd

app = Flask(__name__)
df = pd.read_csv("artifacts/data_ingestion/olist_customers_dataset.csv")
customer_cities = list(df['customer_city'].unique())
customer_states = list(df['customer_state'].unique())
product_categories = list(df['product_category_name_english'].unique())

prediction_pipeline = PredictionPipeline()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        return redirect('/predict')
    
    return render_template('index.html', customer_cities=customer_cities, customer_states=customer_states, product_categories=product_categories)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data as array of inputs
    inputs = [
        request.form['order_status'],
        request.form['payment_type'],
        int(request.form['payment_installments']),
        float(request.form['payment_value']),
        request.form['customer_city'],
        request.form['customer_state'],
        float(request.form['price']),
        float(request.form['freight_value']),
        int(request.form['product_description_length']),
        int(request.form['product_photos_qty']),
        request.form['product_category'],
        int(request.form['Order_purchase_hour']),
        int(request.form['Order_purchase_dayofweek']),
        int(request.form['Time_taken_for_delivery']),
        int(request.form['delivery_delay'])
    ]

    prediction = prediction_pipeline.predict(inputs)

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)