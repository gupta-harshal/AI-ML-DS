import pickle
from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application
ridge_model = pickle.load(open(r"C:\Users\hgp99\OneDrive\Desktop\AI-ML-DS\Regression Projects\Model Training with CV\models\ridge.pkl", 'rb'))
standard_scaler = pickle.load(open(r"C:\Users\hgp99\OneDrive\Desktop\AI-ML-DS\Regression Projects\Model Training with CV\models\scaler.pkl", 'rb'))

# rb means read binary, so we can load the model and scaler


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        WS= float(request.form['WS'])
        Rainfall = float(request.form['Rainfall'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        new_data=standard_scaler.transform([[Temperature, RH, WS, Rainfall, FFMC, DMC, ISI, Classes, Region]])
        result= ridge_model.predict(new_data)

        return render_template('home.html', result=result[0])


    else:
        return render_template('home.html')
    
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)