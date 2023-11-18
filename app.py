from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import pandas as pd
import joblib  # Update import to directly import joblib
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize Flask app
app = Flask(__name__)

def scale(payload):
    """Scales Payload"""
    logger.info(f"Scaling Payload: \n{payload}")
    scaler = StandardScaler().fit(payload.astype(float))
    scaled_adhoc_predict = scaler.transform(payload.astype(float))
    return scaled_adhoc_predict

@app.route("/")
def home():
    html = "<h3>Sklearn Prediction Home</h3>"
    return html

@app.route("/predict", methods=['POST'])
def predict():
    """Performs an sklearn prediction"""
    
    # Logging the input payload
    json_payload = request.json
    logger.info(f"JSON payload: \n{json_payload}")
    inference_payload = pd.DataFrame(json_payload)
    logger.info(f"Inference payload DataFrame: \n{inference_payload}")
    
    # Scale the input
    scaled_payload = scale(inference_payload)
    
    # Load pretrained model as clf
    clf = joblib.load("./model_data/boston_housing_prediction.joblib")
    
    # Get an output prediction from the pretrained model, clf
    prediction = list(clf.predict(scaled_payload))
    
    # Log the output prediction value
    logger.info(f"Prediction: {prediction}")
    
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)  # Specify port=80