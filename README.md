📋 Overview

This project builds a machine learning–based Credit Scoring System that predicts whether a loan applicant is likely to default on a loan based on their financial history.
Using advanced classification algorithms and feature analysis, the model helps financial institutions make accurate, data-driven credit decisions.

📂 Dataset

File: credit.csv
Target Column: loan_default
Feature Examples:

income

loan_amount

debt_to_income_ratio

payment_history_score

num_of_credit_lines

age

⚙️ Features

Automated target column detection

Missing value handling (median/mode imputation)

Label encoding for categorical columns

Feature scaling using StandardScaler

Model comparison (Logistic Regression, Decision Tree, Random Forest)

Evaluation metrics: Precision, Recall, F1-Score, ROC-AUC, and Cross-Validation

Feature importance and confusion matrix visualization

Model persistence using joblib

🧩 Tech Stack
Category	Tools Used
Programming Language	Python
Libraries	Pandas, NumPy, Scikit-learn, Matplotlib
Logging & Model Saving	logging, joblib
Visualization	Matplotlib
Model Types	Logistic Regression, Decision Tree, Random Forest
🚀 How It Works
1️⃣ Load Dataset

The program loads data from credit.csv and checks for missing or invalid values.

2️⃣ Preprocess Data

Cleans and encodes categorical features, fills missing values, and scales numeric data.

3️⃣ Train & Evaluate Models

Three models are trained and compared using stratified 5-fold cross-validation:

Logistic Regression

Decision Tree

Random Forest

4️⃣ Analyze Results

The Random Forest model achieves the highest performance with perfect classification on the test data.
Feature importance and confusion matrix visualizations are generated automatically.

5️⃣ Save Best Model

The best model is saved as:

credit_model.pkl


This allows easy loading and use in real-world applications.

📊 Example Outputs
Feature Importance

Shows which financial features most influence loan default:

Top Predictors:
1. Payment History Score
2. Debt-to-Income Ratio
3. Income

Confusion Matrix

Displays how accurately the model classifies loan defaulters vs non-defaulters.

🧠 Insights

Payment History Score and Debt-to-Income Ratio are the strongest predictors.

The Random Forest Classifier outperformed other models in precision, recall, and F1-score.

Perfect predictions indicate clean data and strong feature separation.

💾 Saving & Loading Model

To save the trained model:

joblib.dump(best_model, "credit_model.pkl")


To load it later:

model = joblib.load("credit_model.pkl")

🧰 Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/yourusername/credit-scoring-model.git
cd credit-scoring-model

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run the Script
python credit_model.py

4️⃣ View Outputs

Feature Importance plot

Confusion Matrix

Model metrics in the console

credit_model.pkl saved in your directory

📈 Future Improvements

Add hyperparameter tuning (GridSearchCV)

Handle imbalanced data using SMOTE or class weights

Deploy as an API using Flask or FastAPI

Create a simple web dashboard for visualization
