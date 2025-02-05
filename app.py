from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()

# Load the pre-trained model
model = joblib.load('modell.pkl')

# Load the same scaler used during training
scaler = joblib.load('scalerr.pkl')  # Save the scaler when training the model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        input_features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])

        # Scale the input using the same scaler
        input_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Convert numeric prediction to class label
        prediction_name = iris.target_names[prediction][0]

        return jsonify({'prediction': prediction_name})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
