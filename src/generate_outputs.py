import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ---------------- LOAD DATA ----------------
df = pd.read_csv("../data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- SMOTE ----------------
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1)
model.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_pred = model.predict(X_test)

# ---------------- CREATE OUTPUT FOLDER ----------------
os.makedirs("../outputs", exist_ok=True)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../outputs/confusion_matrix.png")
plt.close()

# ---------------- CLASSIFICATION REPORT ----------------
report = classification_report(y_test, y_pred)

with open("../outputs/classification_report.txt", "w") as f:
    f.write(report)
# CLASS DISTRIBUTION
df["Class"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.savefig("../outputs/class_distribution.png")
plt.close()

print("Outputs generated successfully!")