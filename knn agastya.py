import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

file_path = r'C:\Users\Admin\Documents\Spotify Most Streamed Songs.csv'
data = pd.read_csv(r"C:\Users\Admin\Downloads\Spotify Most Streamed Songs.csv")

data['streams'] = pd.to_numeric(data['streams'].str.replace(',', ''), errors='coerce')
clean_data = data.dropna(subset=['streams', 'danceability_%', 'energy_%', 'valence_%']).copy()

threshold = clean_data['streams'].median()
clean_data['high_streams'] = (clean_data['streams'] > threshold).astype(int)

X = clean_data[['danceability_%', 'energy_%', 'valence_%']]
y = clean_data['high_streams']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Displaying the first 9 predictions
print("Sample Predictions and Actual Values:")
for i in range(9):
    print(f"Prediction: {y_pred[i]}, Actual: {y_test.iloc[i]}")

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)

