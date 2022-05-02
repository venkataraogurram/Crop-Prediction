import numpy as np
from flask import Flask, request, jsonify, render_template, Markup
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #int_features=[int(x) for x in request.form.values()]
    n=int(request.form['N'])
    p=int(request.form['P'])
    k=int(request.form['K'])
    temp=float(request.form['temperature'])
    humid=float(request.form['humidity'])
    ph=float(request.form['Ph'])
    rain=float(request.form['rainfall'])
    final_features=np.array([[n,p,k,temp,humid,ph,rain]])
    prediction=model.predict(final_features)
    
    output=prediction[0]
    
    return render_template('index.html',prediction_text='Predicted Crop is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=False)
    