import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Chargement simple (on mettra le bon chemin après avoir mis le CSV)
df = pd.read_csv("data/heart_failure.csv")
print("Shape:", df.shape)
print(df.head(3))

# Exemple minimal : adaptera plus tard selon les colonnes réelles
y = df["target"]
X = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
