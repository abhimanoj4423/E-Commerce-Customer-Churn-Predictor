import numpy as np
import pandas as pd
import pickle
import joblib
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request

app=Flask(__name__,template_folder='template')

model = joblib.load(open('dtclassifier.joblib', 'rb'))

encoder = joblib.load(open('Encoder.joblib', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Tenure = float(request.form.get('Tenure'))
    Complain = request.form.get('Complain')
    OrderCount = float(request.form.get('OrderCount'))
    CashbackAmount = float(request.form.get('CashbackAmount'))
    PreferedOrderCat = request.form.get('PreferedOrderCat')
    MaritalStatus = request.form.get('MaritalStatus')

    inputs = pd.DataFrame(np.array([Tenure,Complain, OrderCount, CashbackAmount,PreferedOrderCat,MaritalStatus]).reshape(1, -1), 
                                    columns=['Tenure', 'Complain', 'OrderCount', 'CashbackAmount','PreferedOrderCat','MaritalStatus'])

    #input_processed = encoder.transform(inputs['PreferedOrderCat','MaritalStatus'])
    encoded = encoder.transform(inputs[['PreferedOrderCat','MaritalStatus']])
    encoded_cat_df_new = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    input_processed = pd.concat([inputs.drop(columns=['PreferedOrderCat','MaritalStatus']), encoded_cat_df_new], axis=1)

    prediction = model.predict(input_processed)

    if prediction == 1:
        prediction = 'YES'
    else:
        prediction = 'NO'

    return render_template('predict.html', prediction=prediction, inputs=request.form)

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0',port=5000)