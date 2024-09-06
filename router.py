import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template,request
import os
from utils import preprocess_new

app = Flask(__name__)

model = joblib.load('XGBOOST.pkl')

## Home
@app.route('/')
def home():
    return render_template('index.html')

## predict
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction
        satisfaction_level = float(request.form['satisfaction_level'])
        last_evaluation = float(request.form['last_evaluation'])
        number_project = int(request.form['number_project'])
        average_montly_hours = int(request.form['average_montly_hours'])
        time_spend_company = int(request.form['time_spend_company'])
        Work_accident = request.form['Work_accident']
        promotion_last_5years = request.form['promotion_last_5years']
        salary = request.form['salary']

        # Remmber the Feature Engineering we did
        number_project_per_time_spend_company = number_project / time_spend_company 
        satisfaction_level_per_last_evaluation = satisfaction_level / last_evaluation 
        satisfaction_level_per_number_project = satisfaction_level / number_project 

 
        # Concatenate all Inputs
        X_new = pd.DataFrame({ 'satisfaction_level': [satisfaction_level], 'last_evaluation': [last_evaluation], 'number_project': [number_project],
                              'average_montly_hours': [average_montly_hours], 'average_montly_hours': [average_montly_hours], 'time_spend_company': [time_spend_company], 'Work_accident': [Work_accident],
                              'promotion_last_5years': [promotion_last_5years], 'number_project_per_time_spend_company': [number_project_per_time_spend_company], 'satisfaction_level_per_last_evaluation': [satisfaction_level_per_last_evaluation], 
                              'satisfaction_level_per_number_project': [satisfaction_level_per_number_project],'salary':[salary]
                              })
        
        X_processed = preprocess_new(X_new)
    
        y_pred_new = model.predict(X_processed)
        y_pred_new = '{:.4f}'.format(y_pred_new[0])

        return render_template('predict.html', y_pred_val=y_pred_new)
    else:
        return render_template('predict.html')



## About
@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True)


