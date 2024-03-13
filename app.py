from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors module
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, origins='https://nioushaml-2024-1e2e83037d69.herokuapp.com')  # Initialize CORS with your Flask app and allow requests from the specified origin
model = joblib.load('cad_model.pkl')

@app.route("/", methods=['GET'])
def home():
    return {'message': 'Coronary artery disease prediction'}

@app.route('/predict', methods=['POST'])
def cad_predict():
    try:
        data = request.get_json()
        if 'input' not in data or len(data['input']) != 11:
            return jsonify({"error": "Invalid input data"}), 400
        
        input_values = data['input']
        if not isinstance(input_values, dict):
            return jsonify({"error": "Input data must be a JSON object with named values"}), 400

        # Extract input values from the dictionary and ensure they are in correct order
        input_array = np.array([
            input_values.get("age"),
            input_values.get("sex"),
            input_values.get("chest_pain_type"),
            input_values.get("resting_bps"),
            input_values.get("cholesterol"),
            input_values.get("fasting_blood_sugar"),
            input_values.get("rest_ecg"),
            input_values.get("max_heart_rate"),
            input_values.get("exercise_angina"),
            input_values.get("oldpeak"),
            input_values.get("ST_slope")
        ]).reshape(1, -1)

        pred = model.predict(input_array)
        
        res_val = "Have Coronary Artery Disease" if pred[0] else "Have No Coronary Artery Disease"
        
        return jsonify({'prediction': res_val})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
