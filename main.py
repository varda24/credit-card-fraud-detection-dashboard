import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# from imblearn.over_sampling import SMOTE  # comment this temporarily for fast debug

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print(df.head())

# Split features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Use subset for fast development (optional - remove for full training)
X_train = X_train[:100000]
y_train = y_train[:100000]

# Handle imbalance (optional - comment if needed for fast debug)
# sm = SMOTE(random_state=42)
# X_train, y_train = sm.fit_resample(X_train, y_train)

print("Training data shape:", X_train.shape)

# Model - LogisticRegression (fast + acceptable performance)
model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.show()