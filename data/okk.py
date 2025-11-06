# ========================================
# 1. Import Libraries
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report

# ========================================
# 2. Load Dataset
# ========================================
df = pd.read_csv('winequality-red.csv', sep=';')
print("Dataset shape:", df.shape)
print(df.head())

# ========================================
# 3. Convert Target into Binary Classification
# ========================================
df['quality_label'] = (df['quality'] >= 7).astype(int)
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

# ========================================
# 4. Train/Test Split
# ========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================================
# 5. Feature Scaling
# ========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# 6. Define Models
# ========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ========================================
# 7. Cross-Validation Setup
# ========================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# ========================================
# 8. Evaluate with Cross-Validation
# ========================================
cv_results = {}

for name, model in models.items():
    print(f"\n🔹 Evaluating {name} with 5-Fold Cross-Validation...")
    model.fit(X_train_scaled, y_train)
    
    scores = {}
    for metric, scorer in scoring.items():
        cv_score = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring=scorer)
        scores[metric.capitalize()] = f"{cv_score.mean():.3f} ± {cv_score.std():.3f}"
    
    cv_results[name] = scores

cv_df = pd.DataFrame(cv_results).T
print("\nCross-Validation Results (Mean ± Std):\n")
print(cv_df)

# ========================================
# 9. Evaluate on Test Set
# ========================================
test_results = {}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    test_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }

test_results_df = pd.DataFrame(test_results).T
print("\nTest Set Performance:\n")
print(test_results_df)

# ========================================
# 10. Identify Best Model (F1-score)
# ========================================
best_model_name = test_results_df['F1-score'].idxmax()
print(f"\n✅ Best Model on Test Set: {best_model_name}\n")
print("Detailed Classification Report:\n")
print(classification_report(y_test, models[best_model_name].predict(X_test_scaled)))

# ========================================
# 11. Visualization of Test Results
# ========================================
plt.figure(figsize=(10,6))
test_results_df.plot(kind='bar')
plt.title('Model Performance Comparison (Test Set)', fontsize=16)
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=0)
plt.legend(title='Metric')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

