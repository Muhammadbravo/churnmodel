from flask import Flask,render_template,request
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
import pandas as pd


app = Flask(__name__)


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
	return render_template("index.html",static_url_path='/static')

@app.route('/predict',methods=['POST']) # route to show the predictions in a web UI
def predict():
		try:
			#reading the inputs given by the user
			internetservice = str(request.form['internetservice'])
			onlinebackup = str(request.form['onlinebackup'])
			onlinesecurity = str(request.form['onlinesecurity'])
			contract = str(request.form['contract'])
			paperlessbilling = str(request.form['paperlessbilling'])
			paymentmethod = str(request.form['paymentmethod'])
			tenure = int(request.form['tenure'])
			monthlycharges = float(request.form['monthlycharges'])
			totalcharges = tenure*monthlycharges

			user_inputs = [internetservice,onlinebackup,onlinesecurity,contract,paperlessbilling,paymentmethod,tenure,monthlycharges,totalcharges]

			# load model
			with open('churn-model.bin', 'rb') as f_in:
				dv, model = pickle.load(f_in)
			
			# load inputs and convert to df
			def input_to_df(input_list):
				with open('model_cols.bin', 'rb') as f_in:
					cols = pickle.load(f_in) 
				map_df = dict(zip(cols,input_list))
				input_df = pd.DataFrame([map_df])
				return input_df
			
			# predict
			def predict_single(customer, dv, model):
				X = dv.transform(customer)
				y_pred = model.predict(X)[0]
				y_pred_prob = round(model.predict_proba(X)[:, 1][0],3)
				if y_pred_prob >= 0.5:
					churn = 'Churn'
					return churn, y_pred_prob
				else:
					not_churn = 'Non-Churn'
					return not_churn, y_pred_prob
			
			inputs_df = input_to_df(user_inputs)
			prediction, prop_prediction = predict_single(inputs_df.to_dict(orient='records'), dv, model)

			return render_template('predict.html',prediction= prediction, prop_prediction = round(prop_prediction*100, 2))
		except Exception as e:
			print('The Exception message is: ',e)
			return 'something is wrong'


if __name__ == '__main__':
	app.run(debug=True)