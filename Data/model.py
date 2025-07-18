import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load merged data
data_path = r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\data_merged.csv'
df = pd.read_csv(data_path)

# Feature columns
features = [
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'NAME_CONTRACT_TYPE_x',
    'NAME_FAMILY_STATUS',
    'CNT_CHILDREN',
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT_x'
]

# Remove 'Refused' class from the dataset
df = df[df['NAME_CONTRACT_STATUS'] != 'Refused']
df = df[df['NAME_CONTRACT_STATUS'] != 'Unused offer']

# Drop rows with missing target or feature data
df = df.dropna(subset=['NAME_CONTRACT_STATUS'] + features)

# Prepare features and target
X = df[features]
y = df['NAME_CONTRACT_STATUS']

# Encode categorical features
X = pd.get_dummies(X)

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset stratified by target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=range(len(le.classes_)), target_names=le.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance plot 
#importances = model.feature_importances_
#indices = np.argsort(importances)[::-1]
#features_list = X.columns
#plt.figure(figsize=(10, 6))
#plt.title('Feature Importance from Random Forest')
#plt.bar(range(X.shape[1]), importances[indices], align='center')
#plt.xticks(range(X.shape[1]), [features_list[i] for i in indices], rotation=90)
#plt.tight_layout()
#plt.savefig(r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\feature_importance.png')
#plt.show()

# Testing with New Data - 10 scenarios
new_applications = pd.DataFrame([
    {
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'Y',
        'FLAG_OWN_REALTY': 'N',
        'NAME_CONTRACT_TYPE_x': 'Cash loans',
        'NAME_FAMILY_STATUS': 'Married',
        'CNT_CHILDREN': 1,
        'AMT_INCOME_TOTAL': 100000,
        'AMT_CREDIT_x': 500000
    },
    {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y',
        'NAME_CONTRACT_TYPE_x': 'Cash loans',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'CNT_CHILDREN': 0,
        'AMT_INCOME_TOTAL': 75000,
        'AMT_CREDIT_x': 300000
    },
    {
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y',
        'NAME_CONTRACT_TYPE_x': 'Revolving loans',
        'NAME_FAMILY_STATUS': 'Civil marriage',
        'CNT_CHILDREN': 2,
        'AMT_INCOME_TOTAL': 85000,
        'AMT_CREDIT_x': 400000
    },
    {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'Y',
        'FLAG_OWN_REALTY': 'Y',
        'NAME_CONTRACT_TYPE_x': 'Cash loans',
        'NAME_FAMILY_STATUS': 'Married',
        'CNT_CHILDREN': 3,
        'AMT_INCOME_TOTAL': 120000,
        'AMT_CREDIT_x': 600000
    },
    {
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'Y',
        'FLAG_OWN_REALTY': 'Y',
        'NAME_CONTRACT_TYPE_x': 'Revolving loans',
        'NAME_FAMILY_STATUS': 'Separated',
        'CNT_CHILDREN': 0,
        'AMT_INCOME_TOTAL': 55000,
        'AMT_CREDIT_x': 200000
    },
    {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'N',
        'NAME_CONTRACT_TYPE_x': 'Cash loans',
        'NAME_FAMILY_STATUS': 'Widow',
        'CNT_CHILDREN': 1,
        'AMT_INCOME_TOTAL': 65000,
        'AMT_CREDIT_x': 250000
    },
    {
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'Y',
        'NAME_CONTRACT_TYPE_x': 'Cash loans',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'CNT_CHILDREN': 0,
        'AMT_INCOME_TOTAL': 70000,
        'AMT_CREDIT_x': 320000
    },
    {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'N',
        'NAME_CONTRACT_TYPE_x': 'Cash loans',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'CNT_CHILDREN': 0,
        'AMT_INCOME_TOTAL': 30000,
        'AMT_CREDIT_x': 50000  
    },
    {
        'CODE_GENDER': 'F',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'N',
        'NAME_CONTRACT_TYPE_x': 'Revolving loans',
        'NAME_FAMILY_STATUS': 'Separated',
        'CNT_CHILDREN': 0,
        'AMT_INCOME_TOTAL': 25000,
        'AMT_CREDIT_x': 150000 
    },
    {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'Y',
        'FLAG_OWN_REALTY': 'Y',
        'NAME_CONTRACT_TYPE_x': 'Cash loans',
        'NAME_FAMILY_STATUS': 'Married',
        'CNT_CHILDREN': 4,
        'AMT_INCOME_TOTAL': 40000,
        'AMT_CREDIT_x': 100000  
    }
])

# Preprocess new input: one-hot encode
new_applications_encoded = pd.get_dummies(new_applications)

# Align columns with training data
for col in X_train.columns:
    if col not in new_applications_encoded.columns:
        new_applications_encoded[col] = 0
new_applications_encoded = new_applications_encoded[X_train.columns]

# Predict probabilities
probs = model.predict_proba(new_applications_encoded)

# Find index of 'Approved'
approved_index = list(le.classes_).index('Approved')

# Set threshold
threshold = 0.80

# Print predictions with threshold
output_path = r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\Data\prediction.txt'



with open(output_path, 'w') as f:
    for i, prob_array in enumerate(probs):
        approved_prob = prob_array[approved_index]
        decision = "Approved" if approved_prob >= threshold else "Not Approved"
        probs_str = ', '.join(f"{cls}: {prob:.2f}" for cls, prob in zip(le.classes_, prob_array))
        
        line1 = f"Scenario {i+1} probabilities: {{{probs_str}}}"
        line2 = f"Scenario {i+1} final decision based on threshold {threshold}: {decision}\n"
        
        print(line1)
        print(line2)
        
        f.write(line1 + '\n')
        f.write(line2 + '\n')
print(f"\n Predictions printed to txt file: {output_path}")


import matplotlib.pyplot as plt

# Get class labels
class_labels = le.classes_

# Extract approval probabilities for each scenario
approved_index = list(class_labels).index('Approved')
approved_probs = [prob[approved_index] for prob in probs]


plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), approved_probs, color='skyblue')
plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2.5, label=f'Threshold = {threshold}')
plt.title('Approval Probability for Model Scenarios')
plt.xlabel('Scenario')
plt.ylabel('Probability of Approval')
plt.xticks(range(1, 11))
plt.ylim(0, 2)  # Extended y-axis range
plt.legend()
plt.tight_layout()
plt.savefig(r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\scenario_approval_probs.png')
plt.show()


#Linear Regression
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Prepare variables
df_reg = df[['AMT_INCOME_TOTAL', 'AMT_CREDIT_x']].dropna()
X_reg = df_reg[['AMT_INCOME_TOTAL']]
y_reg = df_reg['AMT_CREDIT_x']

# Fit linear regression model
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

# Print coefficients
print(f"Intercept: {reg_model.intercept_:.2f}")
print(f"Coefficient: {reg_model.coef_[0]:.4f}")

plt.figure(figsize=(8, 6))
sns.regplot(
    x='AMT_INCOME_TOTAL',
    y='AMT_CREDIT_x',
    data=df_reg,
    scatter_kws={'alpha': 0.3, 'color': 'purple'}
)
plt.title('Linear Regression: Income vs Credit Amount')
plt.xlabel('Total Income')
plt.ylabel('Credit Amount')
plt.tight_layout()
plt.savefig(r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\income_credit_regression.png')
plt.show()

from sklearn.metrics import roc_curve, auc

# (Approved vs Not Approved)
if len(le.classes_) == 2:
    y_probs = model.predict_proba(X_test)[:, 1]  # Prob of class 1
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\roc_curve.png')
    plt.show()




#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_log_pred = log_model.predict(X_test)

# Get classification report as string
report_str = classification_report(y_test, y_log_pred, target_names=le.classes_)

# Print to console
print("\nLogistic Regression Classification Report:")
print(report_str)

# Save to file
output_path = r'C:\Users\19564\Documents\CapstoneSR\RuizLoans\logistic_reg_results.txt'
with open(output_path, 'w') as f:
    f.write("Logistic Regression Classification Report:\n")
    f.write(report_str)

print(f"\nClassification report saved to: {output_path}")








