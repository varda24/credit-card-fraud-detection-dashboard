import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os


# Load data
df = pd.read_csv("../data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# create models folder if not exists
os.makedirs("../models", exist_ok=True)
# Model
model = RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/model.pkl")

print("Model saved!")