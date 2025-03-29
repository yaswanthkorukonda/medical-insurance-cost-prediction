import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Load the dataset
df = pd.read_csv('C:/Users/yaswa/IdeaProjects/insurance_prediction_app/insurance.csv')

# Print column names
print("Column names:", df.columns.tolist())

# Check for missing values
print("\nMissing values:")
for col in df.columns:
    print(f"{col}: {df[col].isnull().sum()}")

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in ['sex', 'smoker', 'region']:
    if df[col].dtype == object:
        df[col] = le.fit_transform(df[col])

# Handle numerical columns
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['charges'] = pd.to_numeric(df['charges'], errors='coerce')

X = df.drop(['charges'], axis=1)
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE: {mse}")

# Save the trained model
pickle.dump(model, open('models/rf_tuned.pkl', 'wb'))