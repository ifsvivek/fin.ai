import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (precision_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve,
                             accuracy_score, r2_score, f1_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import warnings

# Configurations
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 0})  # Hide warnings

sns.set_style('whitegrid')

# Loading in the dataset

data = pd.read_csv("data.csv")
data.head()
# Checking the information about each columns to have a basic understanding
data.info()
# This shows a statistical description of the dataset
data.describe()
# identifying the missing values
data.isnull().sum()
dataset = data.drop(['loan_id'], axis = 1)

## Exploartory Data Analysis
dataset.columns = dataset.columns.str.strip()
if 'education' in dataset.columns and 'loan_status' in dataset.columns:
    education_loan_status_counts = dataset.groupby(['education', 'loan_status']).size().unstack()
    plt.figure(figsize=(8, 6))
    ax = education_loan_status_counts.plot(kind='bar', stacked=True)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x+width/2,
                y+height/2,
                '{:.0f}'.format(height),
                horizontalalignment='center',
                verticalalignment='center')
    plt.title('Loan Status by Education Level')
    plt.xlabel('Education Level')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', loc='upper right', labels=['Approved', 'Rejected'], fontsize='small')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    
dataset.columns = dataset.columns.str.strip()
if 'self_employed' in dataset.columns and 'loan_status' in dataset.columns:
    education_loan_status_counts = dataset.groupby(['self_employed', 'loan_status']).size().unstack()
    plt.figure(figsize=(8, 6))
    ax = education_loan_status_counts.plot(kind='bar', stacked=True)
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.text(x+width/2,
                y+height/2,
                '{:.0f}'.format(height),
                horizontalalignment='center',
                verticalalignment='center')
    plt.title('Loan Status by Self_employed')
    plt.xlabel('Self_employed')
    plt.ylabel('Count')
    plt.legend(title='Loan Status', loc='upper right', labels=['Approved', 'Rejected'], fontsize='small')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    
    
loan_status_counts = dataset['loan_status'].value_counts()

plt.figure(figsize=(6, 4))
bars = plt.bar(loan_status_counts.index, loan_status_counts.values, color=['green', 'red'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, yval, ha='center', va='bottom')
plt.title('Loan Status Distribution')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.legend(bars, loan_status_counts.index, title='Loan Status', loc='upper right', fontsize='small')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataset['self_employed'] = le.fit_transform(dataset['self_employed'])
dataset['loan_status'] = le.fit_transform(dataset['loan_status'])
dataset['education'] = le.fit_transform(dataset['education'])

dataset.head()


numeric_columns = dataset.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


df = dataset
# Classification Modeling

# Split dataset
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X.head()
X.describe()
# Create train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Data Scaling

# First we need to know which columns are binary, nominal and numerical
def get_columns_by_category():
    categorical_mask = X.select_dtypes(
        include=['object']).apply(pd.Series.nunique) == 2
    numerical_mask = X.select_dtypes(
        include=['int64', 'float64']).apply(pd.Series.nunique) > 5

    binary_columns = X[categorical_mask.index[categorical_mask]].columns
    nominal_columns = X[categorical_mask.index[~categorical_mask]].columns
    numerical_columns = X[numerical_mask.index[numerical_mask]].columns

    return binary_columns, nominal_columns, numerical_columns

binary_columns, nominal_columns, numerical_columns = get_columns_by_category()

# Now we can create a column transformer pipeline

transformers = [('binary', OrdinalEncoder(), binary_columns),
                ('nominal', OneHotEncoder(), nominal_columns),
                ('numerical', StandardScaler(), numerical_columns)]

transformer_pipeline = ColumnTransformer(transformers, remainder='passthrough')

# Starified k cross validation
Kfold = StratifiedKFold(n_splits=5)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Lets stack up our classifiers:

RANDOM_STATE = 42
classifiers = [LogisticRegression(max_iter=70, solver='sag', random_state=RANDOM_STATE),
               DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
               SVC(C=2, kernel='linear', random_state=RANDOM_STATE),
               RandomForestClassifier(
                   max_depth=7, min_samples_split=5, min_samples_leaf=5, random_state=RANDOM_STATE),
               LGBMClassifier(num_leaves=300,max_depth= 8, n_estimators=1000, learning_rate=0.04,
                              min_data_in_leaf = 300),
               AdaBoostClassifier(DecisionTreeClassifier(
                   max_depth=1, random_state=RANDOM_STATE), random_state=RANDOM_STATE),
               GradientBoostingClassifier(
                   learning_rate=0.005, n_estimators=30, random_state=RANDOM_STATE),
               KNeighborsClassifier(),
               GaussianNB(var_smoothing=1e-2)
               ]

classifiers_names = ['Logistic Regression',
                     'Decision Tree Classifier',
                     'Support Vector Machine',
                     'Random Forest Classifier',
                     'LGBMClassifier',
                     'AdaBoost Classifier',
                     'Gradient Boosting Classifier',
                     'K Neighbors Classifier',
                     'Gaussian Naive Bayes']

pipelines = [Pipeline([('transformer', transformer_pipeline), (classifier_name, classifier)])
             for classifier_name, classifier in zip(classifiers_names, classifiers)]

# Cross Validation.

def cv_fit_models():
    train_acc_results = []
    cv_scores = {classifier_name: [] for classifier_name in classifiers_names}
    for classifier_name, pipeline in zip(classifiers_names, pipelines):
        cv_score = cross_validate(pipeline,
                                  X_train,
                                  y_train,
                                  scoring=scoring,
                                  cv=Kfold,
                                  return_train_score=True,
                                  return_estimator=True)

        train_accuracy = cv_score['train_acc'].mean() * 100

        train_acc_results.append(train_accuracy)
        cv_scores[classifier_name].append(cv_score)

    return np.array(train_acc_results), cv_scores

scoring = {'acc': 'accuracy'}

results, folds_scores = cv_fit_models()

# Pick the best fold for each model according to the highest test accuracy:

def pick_best_estimator():
    best_estimators = {classifier_name: [] for classifier_name in classifiers_names}
    for key, model in folds_scores.items():
        best_acc_idx = np.argmax(model[0]['test_acc'])
        best_model = model[0]['estimator'][best_acc_idx]
        best_estimators[key].append(best_model)
    return best_estimators

# Now we finally can get the accuracy scores of each best fold
# and at the same time get their precision & recall scores:

def gather_metrics_scores():
    test_accs, precisions, recalls = [], [], []
    for estimator_val in best_estimators.values():
        estimator = estimator_val[0]
        y_pred = estimator.predict(X_test)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        score = estimator.score(X_test, y_test)

        test_accs.append(score)
        precisions.append(precision)
        recalls.append(recall)

    scores = {'test_acc': np.array(test_accs),
              'precision': np.array(precisions),
              'recall': np.array(recalls)}

    return scores

scores = gather_metrics_scores()

# Plot metrics

def plot_train_test_accuracy(df):
    _, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
    sns.barplot(data=df, x='train_accuracy',
                y='Model', orient='h', ax=ax[0],
               order=df.sort_values('train_accuracy',ascending = False).Model)
    ax[0].set_xlim([0, 100])
    sns.barplot(data=df, x='test_accuracy',
                y='Model', orient='h', ax=ax[1],
               order=df.sort_values('test_accuracy',ascending = False).Model)
    ax[1].set_xlim([0, 100])
    
    
results_df = pd.DataFrame({'Model': classifiers_names,
                           'train_accuracy': results,
                           'test_accuracy': scores['test_acc'] * 100,
                           'test_precision': scores['precision'] * 100,
                           'test_recall': scores['recall'] * 100})

results_df

plot_train_test_accuracy(results_df)

def plot_precision_recall(df):
    _, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
    sns.barplot(data=df, x='test_precision',
                y='Model', orient='h', ax=ax[0],
               order=df.sort_values('test_precision',ascending = False).Model)
    ax[0].set_xlim([0, 100])
    sns.barplot(data=df, x='test_recall',
                y='Model', orient='h', ax=ax[1],
               order=df.sort_values('test_recall',ascending = False).Model)
    ax[1].set_xlim([0, 100])
    
plot_precision_recall(results_df)

import joblib
clf = RandomForestClassifier(
                   max_depth=7, min_samples_split=5, min_samples_leaf=5, random_state=42)

# Train the model
clf.fit(X_train, y_train)


# Save the model to a file
joblib.dump(clf, 'random_forest_model.pkl')

print("Model saved successfully.")

import joblib

# Load the model from the file
loaded_model = joblib.load('random_forest_model.pkl')

print("Model loaded successfully.")

y_test.head()