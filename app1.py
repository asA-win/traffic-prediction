from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Load dataset
data = pd.read_csv("traffic1.csv")  # Replace "traffic1.csv" with your actual dataset filename

# Preprocess data
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d-%m-%Y %H:%M')  # Adjust the date format if needed
data['Hour'] = data['DateTime'].dt.hour
X = data[['Hour', 'Junction']]
y = data['Vehicles']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        date = request.form['date']
        time = request.form['time']
        junction = int(request.form['junction'])
        
        datetime_obj = datetime.strptime(date + ' ' + time, '%Y-%m-%d %H:%M')
        hour = datetime_obj.hour
        
        # Make prediction for all junctions
        predictions = []
        for j in range(1, 5):
            predicted_vehicles = model.predict([[hour, j]])
            traffic_flow = classify_traffic_flow(predicted_vehicles[0])  # You need to define this function
            predictions.append((j, predicted_vehicles[0], traffic_flow))
        
        # Find the junction with the lowest predicted traffic
        best_junction = min(predictions, key=lambda x: x[1])
        
        return render_template('result.html', predictions=predictions, best_junction=best_junction)

def classify_traffic_flow(predicted_vehicles):
    # You need to define your logic for classifying traffic flow based on the predicted number of vehicles
    # This can be simple rules or a machine learning model trained for traffic flow classification
    # For example:
    if predicted_vehicles < 10:
        return 'Low traffic flow'
    elif predicted_vehicles < 20:
        return 'Moderate traffic flow'
    else:
        return 'High traffic flow'

if __name__ == '__main__':
    app.run(debug=True)
