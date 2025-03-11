import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset & trained model
df = pd.read_csv("data.csv")
model = joblib.load("rf_model.pkl")

# Drop unnecessary columns
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
df.drop(columns=["id"], inplace=True)

# Separate features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Get feature importance from the model
importances = model.feature_importances_
feature_names = X.columns

# Create DataFrame for feature importance
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Print top features
print(feature_importance_df.head(15))  # Show top 15 features

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["Feature"][:15], feature_importance_df["Importance"][:15], color="teal")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Important Features for Breast Cancer Detection")
plt.gca().invert_yaxis()
plt.show()
