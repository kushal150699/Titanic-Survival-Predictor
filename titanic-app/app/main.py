import os
import pickle
import numpy as np

from flask import Flask, render_template, request
app = Flask(__name__)


def Predictor(predict_list):

    to_predict = np.array(predict_list).reshape(1,7)
    filename = 'Model/model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(to_predict)
    return result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods = ['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        print(predict_list)         # for testing
        predict_list = list(predict_list.values())
        print(predict_list)         # for testing
        predict_list = list(map(int, predict_list))
        print(predict_list)
        result = Predictor(predict_list)
        print(result)               # for testing
        
        if int(result) == 1: 
            prediction = 'Passenger Survived!'
        elif int(result) == 0: 
            prediction = "Passenger didn't survive."
                
        return render_template('prediction.html', prediction = prediction)


if __name__ == '__main__':
    app.run()
