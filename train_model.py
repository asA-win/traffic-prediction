import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import regularizers
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("traffic.csv")

# Data preprocessing
data['DateTime'] = pd.to_datetime(data['DateTime'])
data['Hour'] = data['DateTime'].dt.hour
data['DayOfWeek'] = data['DateTime'].dt.dayofweek
data['DayOfMonth'] = data['DateTime'].dt.day
data['Month'] = data['DateTime'].dt.month
data.drop(columns=['DateTime', 'ID'], inplace=True)

# Feature engineering: Handle categorical variables if needed

# Splitting data into features (X) and target variable (y)
X = data.drop(columns=['Vehicles'])
y = data['Vehicles']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dropout(0.2))  # Adding dropout for regularization
model.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.01)))  # L2 regularization
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_data=(X_test_lstm, y_test), verbose=2)

# Plot training and validation loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model
# Evaluate the model
train_mse = model.evaluate(X_train_lstm, y_train, verbose=0)
test_mse = model.evaluate(X_test_lstm, y_test, verbose=0)
print(f'Train MSE: {train_mse:.3f}, Test MSE: {test_mse:.3f}')

# Save the model
model.save('trained_lstm_modelll.h5')
