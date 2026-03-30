from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("mind_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    sleep = float(request.form['sleep'])
    work = float(request.form['work'])
    stress = float(request.form['stress'])
    social = float(request.form['social'])
    
    prediction = model.predict([[sleep, work, stress, social]])
    
    return render_template("index.html", result=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)