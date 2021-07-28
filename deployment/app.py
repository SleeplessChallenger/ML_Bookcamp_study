from flask import Flask, request, jsonify
from pickle import dump, load
import numpy as np


app = Flask('churn_prediction')


def predict_single(customer, dv, model):
	X = dv.transform([customer])
	y_pred = model.predict_proba(X)[:, 1]
	return y_pred[0]

with open('churn-model.bin',  'rb') as churn_model_des:
	dv, model = load(churn_model_des)


@app.route('/predict', methods=['POST'])
def get_prediction():
	data = request.get_json()
	prediction = predict_single(data, dv, model)
	churn = prediction >= 0.5

	result = {
		'prob.': round(prediction, 3),
		'churn': bool(churn)
	}

	return jsonify(result)


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=8085)
