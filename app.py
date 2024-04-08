from flask import Flask, request, jsonify
import mysql.connector
from joblib import load
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS with your specific origin
model = load('cad_model.pkl')  # Load your ML model

# Connect to MySQL database
db = mysql.connector.connect(
    host="127.0.0.1",
    port="3306",
    user="root",
    password="password",
    database="cad"
)

@app.route('/insert_patient', methods=['POST'])
def insert_patient():
    # Get data from the request
    data = request.json

    # Assuming your patient data structure, adjust the fields accordingly
    patient_id = data.get('patient_id')
    age = data.get('age')
    sex = data.get('sex')
    chest_pain_type = data.get('chest_pain_type')
    resting_bps = data.get('resting_bps')
    cholesterol = data.get('cholesterol')
    # Add more fields as per your model's requirements

    # Insert data into MySQL table
    cursor = db.cursor()
    query = "INSERT INTO patient (patient_id, age, sex, chest_pain_type, resting_bps, cholesterol) VALUES (%s, %s, %s, %s, %s, %s)"
    values = (patient_id, age, sex, chest_pain_type, resting_bps, cholesterol)
    cursor.execute(query, values)
    db.commit()
    cursor.close()
    return jsonify({'message': 'Patient data inserted successfully'}), 201

@app.route('/predict', methods=['POST'])
def predict():
    
    # Get patient_id from the request
    patient_id = request.json.get('patient_id')

    # Retrieve patient information from the database
    cursor = db.cursor()
    query = "SELECT * FROM patient WHERE patient_id = %s"
    cursor.execute(query, (patient_id,))
    patient_data = cursor.fetchone()
    cursor.close()

    if not patient_data:
        return jsonify({'error': 'Patient not found'}), 404

    # Prepare input features for prediction based on your model's requirements
    features = np.array([patient_data[1], patient_data[2], patient_data[3], patient_data[4], patient_data[5]]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
