from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


appEd = Flask(__name__)
Bootstrap(appEd)


@appEd.route('/')
def index():
	return render_template('index.html')

@appEd.route('/predict', methods=['POST'])
def predict():
	#df= pd.read_csv("data/names_dataset.csv")
	# Features and Labels
	#df_X = df.name
	#df_Y = df.sex

    # Vectorization
	#corpus = df_X
	#cv = CountVectorizer()
	#X = cv.fit_transform(corpus)

	# Loading our ML Model
	SVM_model = open("models/model_SVM.pk","rb")
	mdl = joblib.load(SVM_model)

	# Receives the input query from form
	if request.method == 'POST':
		namequery = request.form['namequery']
		#data = [namequery]
		#vect = cv.transform(data).toarray()
		my_prediction = mdl.predict([namequery])
	return render_template('results.html', prediction = my_prediction)


if __name__ == '__main__':
	appEd.run(debug=True)
