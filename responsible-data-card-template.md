import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
import joblib

# Load dataset
df = pd.read_csv('customer-churn-training.csv')
X = df.drop(columns=['customer_id', 'churned'])
y = df['churned']

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Feature Groups
numeric_features = ['tenure_months', 'support_tickets', 'monthly_spend_inr', 'last_login_days']
categorical_features = ['plan_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))]), categorical_features)
    ])

# 1. Train 4 Distinct Model Architectures
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42)
}

model_results = []
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else [0]*len(y_test)
    
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    
    model_results.append({'Model': name, 'F1-Score': f1, 'ROC-AUC': auc})

results_df = pd.DataFrame(model_results)
print("--- Model Comparison Table ---")
print(results_df.to_string(index=False))

# 2. Hyperparameter Optimization using GridSearchCV & Stratified K-Fold
param_grid = {
    'classifier__n_estimators': [10, 50, 100],
    'classifier__max_depth': [None, 3, 5],
    'classifier__min_samples_split': [2, 4]
}

rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring='f1')
grid_search.fit(X_train, y_train)

print("\n--- Hyperparameter Tuning Results ---")
print("Best CV F1 Score:", grid_search.best_score_)
print("Best Hyperparameters:", grid_search.best_params_)

# 3. Champion Model Serialization
champion_model = grid_search.best_estimator_
joblib.dump(champion_model, 'champion_churn_model.joblib')
print("\nChampion model successfully serialized to 'champion_churn_model.joblib'")
