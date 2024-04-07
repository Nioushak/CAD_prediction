from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3

app = Flask(__name__)
CORS(app, origins='https://nioushaml-2024-1e2e83037d69.herokuapp.com')
model = joblib.load('cad_model.pkl')

# Function to get patient data from the database
def get_patient_data(patient_id):
    conn = sqlite3.connect('patients.db')  # Connect to your SQLite database
    cursor = conn.cursor()
    query = "SELECT age, sex, chest_pain_type, resting_bps, cholesterol, fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope FROM patients WHERE patient_id = ?"
    cursor.execute(query, (patient_id,))
    data = cursor.fetchone()  # Assuming patient_id is unique and only one record is returned
    conn.close()
    return data

@app.route("/", methods=['GET'])
def home():
    return {'message': 'Coronary artery disease prediction'}

@app.route('/predict', methods=['POST'])
def cad_predict():
    try:
        data = request.get_json()
        if 'patient_id' not in data:
            return jsonify({"error": "Patient ID is required"}), 400
        
        patient_id = data['patient_id']
        patient_data = get_patient_data(patient_id)
        if not patient_data:
            return jsonify({"error": "Patient not found"}), 404

        # Convert patient data to numpy array
        input_array = np.array(patient_data).reshape(1, -1)

        pred = model.predict(input_array)
        
        res_val = "Have Coronary Artery Disease" if pred[0] else "Have No Coronary Artery Disease"
        
        return jsonify({'prediction': res_val})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
