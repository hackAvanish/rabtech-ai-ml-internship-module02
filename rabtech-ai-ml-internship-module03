import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

# 1. Load Dataset
df = pd.read_csv('customer-churn-training.csv')

# Separate features and target label
X = df.drop(columns=['customer_id', 'churned'])
y = df['churned']

# 2. Train/Test Split BEFORE any transformations (Prevents Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Define feature groups
numeric_features = ['tenure_months', 'support_tickets', 'monthly_spend_inr', 'last_login_days']
categorical_features = ['plan_type']

# 3. Build Sub-Pipelines for Numerical and Categorical Data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Combine into a unified ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Fit Preprocessor on Training Data Only & Transform Both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Extract transformed feature names for analysis
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
all_features = numeric_features + encoded_cat_features

# 5. Feature Importance & Correlation Analysis
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_processed, y_train)

feature_importances = pd.Series(rf_model.feature_importances_, index=all_features)
print("--- Tree-Based Feature Importances ---")
print(feature_importances.sort_values(ascending=False))

# Mutual Information Scores
mi_scores = mutual_info_classif(X_train_processed, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=all_features)
print("\n--- Mutual Information Scores ---")
print(mi_series.sort_values(ascending=False))
