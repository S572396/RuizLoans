#### Ruiz-Capstone-Loan Approval Project

# A Model for Bank Loan Approval Analysis and Predictions

## Introuction
Banks have to make decisions everyday regarding to approve a customers loan or deny a loan. But how is this done? What are the main factors involved that decides who gets approved while others denied? I was in banking many years and hope to explore the answers with this project with the interesting analysis involved. The authors from 

## Abstract

This project will work with data from a csv file, that will be cleaned and analyzed. A model will be trained,tested, and deployed to make predictions for customer bank loan approvals.

## {Goals of this Research} 
The goal will be to analyze csv data for bank customers and develop a model to predict loan approvals.

## Dataset from:
https://www.kaggle.com/code/gauravduttakiit/risk-analytics-in-banking-financial-services-1/input 

## Project Steps: 1. select topic, 2. select data set, 3. clean data set, 4. convert or transform data if needed, 5. train and develop a model, evaluate the model, Update and Adjust as needed.

### Requirements:
matplotlib==3.8.0
numpy==1.26.2
pandas==2.1.3
scikit-learn==1.3.0
seaborn==0.13.0
notebook==7.0.6

## Acitvate Virutal Environment
py -m venv .venv
.venv\Scripts\Activate
py -m pip install -r requirements.txt



### Section 1. Load and Explore the Data
- 1.1 Load the dataset and display the first 10 rows.
- 1.2 Check for missing values and display summary statistics.
- (removed columns and labels not needed)


### Section 2. Feature Selection and Justification
- 2.1 Choose two input features for predicting the target.
- Linear Regression eliminated, did not show strong correlations of features.


### Section 3. Train a  Model and Process for Exploratory Analysis
loan _model_workflow.ipynb
model.py

Random Forest Selected
-RandomForestClassifier(random_state=42)
-Features were encoded using one-hot encoding, and the target was label-encoded.
-Evalluated using classification report, confusion matrix, and feature importance plot

Logistic Regression Selected
-Explored coefficients for each feature.
-Trained using LogisticRegression(max_iter=1000).
-Vvaluated using F1-score, precision, recall, and ROC AUC score.

Train-Test Split
-Data was split into 80% training and 20% testing
-Passed through the Model
-Probabilities for loan approval were calculated and set
 at 80 percent threshold
-Results were exported to a text file

