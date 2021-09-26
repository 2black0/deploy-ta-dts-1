from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

loaded_model = joblib.load('model.sav')

@app.route('/')
def index():
    return render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        list_to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, list_to_predict_list))
        
        #to_predict = np.array(to_predict_list).reshape(-1, 1)
        #esult = loaded_model.predict(to_predict)
        result = round(float(ValuePredictor(to_predict_list)), 2)
        #result = round(result, 2)
        return render_template("index.html", hasil=result, dana=list_to_predict_list)

if __name__ == '__main__':
    app.run(debug=True)