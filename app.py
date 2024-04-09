# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the trained LSTM model
lstm_model = load_model('trained_lstm_model.h5')

# Function to preprocess input data
def preprocess_input(data):
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['Hour'] = data['DateTime'].dt.hour
    data['DayOfWeek'] = data['DateTime'].dt.dayofweek
    data['DayOfMonth'] = data['DateTime'].dt.day
    data['Month'] = data['DateTime'].dt.month
    data.drop(columns=['DateTime'], inplace=True)
    return data

# Interpretation function
def interpret_traffic_flow(predicted_vehicles):
    if predicted_vehicles < 10:
        return "Light traffic flow"
    elif 10 <= predicted_vehicles < 20:
        return "Moderate traffic flow"
    elif 20 <= predicted_vehicles < 30:
        return "Heavy traffic flow"
    else:
        return "Very heavy traffic flow"

# Function to predict vehicles
def predict_vehicles(input_data):
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    input_data_reshaped = np.reshape(input_data_scaled, (input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))
    predicted_vehicles = lstm_model.predict(input_data_reshaped)
    return predicted_vehicles[0][0]  # Extract the predicted value from the numpy array

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    date_time = pd.to_datetime(request.form['date'] + ' ' + request.form['time'])
    junction = int(request.form['junction'])
    
    # Prepare input data
    input_data = pd.DataFrame({'DateTime': [date_time], 'Junction': [junction]})
    input_data = preprocess_input(input_data)
    
    # Debug prints for input data
    print("User Input:")
    print(input_data)
    
    # Make prediction
    predicted_vehicles = predict_vehicles(input_data)
    
    # Debug print for predicted vehicles
    print("Predicted Vehicles:", predicted_vehicles)
    
    # Interpret traffic flow
    traffic_flow = interpret_traffic_flow(predicted_vehicles)
    
    # Debug print for traffic flow
    print("Traffic Flow:", traffic_flow)
    
    # Render result template with prediction and traffic flow
    return render_template('result.html', predicted_vehicles=predicted_vehicles, traffic_flow=traffic_flow)

if __name__ == '__main__':
    app.run(debug=True)
