# wine_quality_explainability.py
# Visualize SHAP and Permutation Feature Importance for wine quality prediction

import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# === Load dataset ===
data = pd.read_csv("winequality-red.csv", sep=';')

# === Split features and target ===
X = data.drop("quality", axis=1)
y = data["quality"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Train Random Forest ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === SHAP Analysis ===
print("Calculating SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global importance summary
shap.summary_plot(shap_values, X_test, show=True)
plt.title("SHAP Summary Plot")

# === Permutation Feature Importance ===
print("Computing permutation feature importance...")
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(8, 6))
plt.barh(X_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Decrease in Model Performance")
plt.title("Permutation Feature Importance")
plt.tight_layout()

plt.show()
