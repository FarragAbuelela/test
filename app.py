from flask import Flask,request, jsonify
from flask_restful import Resource, Api
import joblib
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
#
CORS(app)
# creating an API object
api = Api(app)

#prediction api call 'F:/v2 machine learning/model 1/deploy api machine learning/
model = joblib.load(open('Farrag Model V2.pkl','rb'))


@app.route('/')
def home():
    return 'players rating api 😊'

@app.route("/predict",methods=["post"])
def predict():
    rates = request.json
    
    quary_df = pd.DataFrame(rates)
    predection = model.predict(quary_df)
    return jsonify(list(predection))
    

if __name__ == '__main__':
    app.run(debug=True)