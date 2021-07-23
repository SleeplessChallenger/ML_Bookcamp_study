from pickle import dump, load
import numpy as np


def predict_single(customer, dv, model):
	X = dv.transform([customer])
	y_pred = model.predict_proba(X)[:, 1]
	return y_pred[0]

with open('churn-model.bin',  'rb') as churn_model_des:
	dv, model = load(churn_model_des)

customer = {
    'customerid': '8879-zkjof',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 3320.75,
}

pred = predict_single(customer, dv, model)

print(f"{round(pred, 3)}")

if pred >= 0.5:
	print('Churn')
else:
	print('No churn')
