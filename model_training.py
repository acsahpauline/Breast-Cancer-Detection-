import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("data.csv")  # Ensure 'data.csv' is in your folder

# Step 2: Preprocessing
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df.drop(columns=["id"], inplace=True)

# Step 3: Split data
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# Step 6: Save the trained model
joblib.dump(rf_model, "rf_model.pkl")
print("✅ Model saved as rf_model.pkl")
