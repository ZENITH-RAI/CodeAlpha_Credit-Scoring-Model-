"""
Pro-Level Credit Scoring Model
Dataset: credit.csv
Target: loan_default
Features: financial data (income, debts, credit history, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ------------------------------
# 1️⃣ Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------
# 2️⃣ Load Dataset
# ------------------------------
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Dataset loaded successfully: {filepath}")
        return df
    except FileNotFoundError:
        logging.error(f"{filepath} not found!")
        raise

# ------------------------------
# 3️⃣ Preprocess Dataset
# ------------------------------
def preprocess_data(df, target_keywords=['default']):
    # Drop rows with all missing values
    df.dropna(how="all", inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Detect target column automatically
    target_col = None
    for col in df.columns:
        if any(k in col.lower() for k in target_keywords):
            target_col = col
            break

    if not target_col:
        raise ValueError("Target column not found! Include 'default' in the target column name.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    logging.info(f"Target column detected: {target_col}")
    return X, y

# ------------------------------
# 4️⃣ Train & Evaluate Models
# ------------------------------
def train_evaluate_models(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    for name, model in models.items():
        # Pipeline (scaling + model)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Cross-validation F1-score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "CV F1-Mean": cv_scores.mean()
        })

        logging.info(f"✅ {name} trained. Test F1: {f1:.2f}, CV F1: {cv_scores.mean():.2f}")

    results_df = pd.DataFrame(results)
    logging.info("\nModel Comparison:\n" + str(results_df))
    return results_df, models['Random Forest'], X_train_scaled, X_test_scaled, y_train, y_test

# ------------------------------
# 5️⃣ Feature Importance
# ------------------------------
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances, color="skyblue")
    plt.yticks(range(len(importances)), feature_names)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()

# ------------------------------
# 6️⃣ Confusion Matrix
# ------------------------------
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Value labels
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

# ------------------------------
# 7️⃣ Main Execution
# ------------------------------
if __name__ == "__main__":
    df = load_data("credit_data.csv")
    X, y = preprocess_data(df)
    results_df, best_model, X_train_scaled, X_test_scaled, y_train, y_test = train_evaluate_models(X, y)

    # Feature importance
    plot_feature_importance(best_model, X.columns)

    # Confusion matrix
    y_pred_best = best_model.predict(X_test_scaled)
    plot_confusion_matrix(y_test, y_pred_best)

    # Save best model
    joblib.dump(best_model, "credit_model.pkl")
    logging.info("✅ Best model saved as credit_model.pkl")
