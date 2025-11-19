import joblib
import numpy as np

# Charger le modèle sauvegardé
bundle = joblib.load("model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["features"]

print("✔ Modèle chargé :", model)

# Exemple d’utilisation locale (facultatif)
# Remplace par une vraie ligne du dataset si tu veux tester
example = np.array([50, 1, 168, 70, 120, 80, 1, 1, 0, 1, 0, 0]).reshape(1, -1)

example_scaled = scaler.transform(example)
prediction = model.predict(example_scaled)

print("Prédiction :", prediction[0])
