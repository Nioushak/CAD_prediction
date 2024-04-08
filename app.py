from flask import Flask, request, jsonify
import mysql.connector
from joblib import load
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)   


# Connect to MySQL database
db = mysql.connector.connect(
    host="127.0.0.1",
    port="3306",
    user="root",
    password="password",
    database="cad"
)

model = load('cad_model.pkl')  

@app.route('/insert_patient', methods=['POST'])
def insert_patient():
    # Get data from the request
    data = request.json

    
    patient_id = data.get('patient_id')
    age = data.get('age')
    sex = data.get('sex')
    chest_pain_type = data.get('chest_pain_type')
    resting_bps = data.get('resting_bps')
    cholesterol = data.get('cholesterol')
    fasting_blood_sugar =  data.get('fasting_blood_sugar')
    rest_ecg = data.get('rest_ecg')
    max_heart_rate = data.get('max_heart_rate')
    exercise_angina = data.get('exercise_angina')
    oldpeak = data.get('oldpeak')
    ST_slope = data.get('ST_slope')
    

    # Insert data into MySQL table
    cursor = db.cursor()
    query = "INSERT INTO patient (patient_id, age, sex, chest_pain_type, resting_bps, cholesterol, fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope) VALUES (%s, %s, %s, %s, %s, %s,%s, %s, %s, %s, %s, %s)"
    values = (patient_id, age, sex, chest_pain_type, resting_bps, cholesterol, fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope )
    cursor.execute(query, values)
    db.commit()
    
    cursor.close()
    return jsonify({'message': 'Patient data inserted successfully'}), 201

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json
    # Get patient_id from the request
    patient_id = data.get('patient_id')

    # Retrieve patient information from the database
    cursor = db.cursor()
    query = "SELECT * FROM patient WHERE patient_id = %s"
    cursor.execute(query, (patient_id,))
    patient_data = cursor.fetchone()
    cursor.close()

    if not patient_data:
        return jsonify({'error': 'Patient not found'}), 404

    # Prepare input features for prediction
    features = np.array([
        
        patient_data[0],  # age
        patient_data[1],  # sex
        patient_data[2],  # chest_pain_type
        patient_data[3],  # resting_bps
        patient_data[4],  # cholesterol
        patient_data[5], # fasting_blood_sugar
        patient_data[6],  # rest_ecg
        patient_data[7],  # max_heart_rate 
        patient_data[8],  #exercise_angina
        patient_data[9],  #oldpeak
        patient_data[10],   #ST_slope
]).reshape(1, -1)

    
    # Make prediction
    prediction = int(model.predict(features)[0])
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
