import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
# Replace the path below with the actual path to the CSV file on your local machine
file_path = r'C:\Users\Admin\Documents\Spotify Most Streamed Songs.csv'
data = pd.read_csv(r"C:\Users\Admin\Downloads\Spotify Most Streamed Songs.csv")

# Clean and preprocess the data
data['streams'] = pd.to_numeric(data['streams'].str.replace(',', ''), errors='coerce')
clean_data = data.dropna(subset=['streams', 'danceability_%', 'energy_%', 'valence_%'])

# Select features and target
X = clean_data[['danceability_%', 'energy_%', 'valence_%']]
y = clean_data['streams']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and error calculation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


