from flask import Flask, request, jsonify
from joblib import load
from flask_cors import CORS
import numpy as np
import mysql.connector
import os
import urllib.parse

app = Flask(__name__)
CORS(app)

# Load model
model = load('cad_model.pkl')
  
def get_db_connection():
    # Retrieve the database URL from the environment variable
    database_url = os.environ.get('DATABASE_URL', 'mysql://fgnt171t3pb9rbjx:m8ckuyqp25350bpm@pqxt96p7ysz6rn1f.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/mmnl46hxhxuoxi6c')
    url = urllib.parse.urlparse(database_url)

    # Connect to the MySQL database using connection details from the URL
    connection = mysql.connector.connect(
        host=url.hostname,
        user=url.username,
        password=url.password,
        database=url.path[1:],  # Skip the leading '/'
        port=url.port or 3306
    )

    # Check if the patient table exists, if not, create it
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES LIKE 'patient'")
    if not cursor.fetchone():
        create_table_query = """
        CREATE TABLE patient (
            patient_id INT AUTO_INCREMENT PRIMARY KEY,
            age INT,
            sex VARCHAR(10),
            chest_pain_type VARCHAR(50),
            resting_bps INT,
            cholesterol INT,
            fasting_blood_sugar VARCHAR(10),
            rest_ecg VARCHAR(50),
            max_heart_rate INT,
            exercise_angina VARCHAR(10),
            oldpeak FLOAT,
            ST_slope VARCHAR(10)
        )
        """
        cursor.execute(create_table_query)
        connection.commit()

    return connection

@app.route('/insert_patient', methods=['POST'])
def insert_patient():
    data = request.json
    # Extract data from request
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

    db = None
    cursor = None
    try:
        db = get_db_connection()
        cursor = db.cursor()
        query = ("INSERT INTO patient (patient_id, age, sex, chest_pain_type, resting_bps, "
                 "cholesterol, fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina, "
                 "oldpeak, ST_slope) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        values = (patient_id, age, sex, chest_pain_type, resting_bps, cholesterol, fasting_blood_sugar,
                  rest_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope)
        cursor.execute(query, values)
        db.commit()
        return jsonify({'message': 'Patient data inserted successfully'}), 201
    except mysql.connector.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    patient_id = data.get('patient_id')

    try:
        db = get_db_connection()
        cursor = db.cursor()
        query = "SELECT age, sex, chest_pain_type, resting_bps, cholesterol, fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina, oldpeak, ST_slope FROM patient WHERE patient_id = %s"
        cursor.execute(query, (patient_id,))
        patient_data = cursor.fetchone()
        if not patient_data:
            return jsonify({'error': 'Patient not found'}), 404

        features = np.array(patient_data).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({'prediction': int(prediction)})
    except mysql.connector.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        db.close()

if __name__ == '__main__':
    app.run(debug=True)
