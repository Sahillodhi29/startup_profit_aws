from flask import Flask , request ,render_template ,jsonify
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

#import pickle model files 

regressor_model=pickle.load(open('Models/Regressor.pkl' , 'rb'))
scaler_model=pickle.load(open('Models/Scaler.pkl' , 'rb'))



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/startup', methods=['GET','POST'])

def predict_profit():

    if request.method=='POST':
        RnD_Spend = float(request.form.get('RnD_Spend'))
        Administration=float(request.form.get('Administration'))
        Marketing_Spend=float(request.form.get('Marketing_Spend'))

        new_data_scaled=scaler_model.transform([[RnD_Spend,Administration,Marketing_Spend]])
        result=regressor_model.predict(new_data_scaled)

        return render_template('home.html' , result=result[0])

        
    else:
        return render_template('home.html')

    


if __name__=="__main__":
    app.run(host="0.0.0.0")
