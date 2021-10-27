#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


# In[3]:


get_ipython().system('pip install imblearn')


# In[4]:


from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced


# # Read the CSV and Perform Basic Data Cleaning

# In[5]:


# https://help.lendingclub.com/hc/en-us/articles/215488038-What-do-the-different-Note-statuses-mean-

columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

target = ["loan_status"]


# In[9]:


# Load the data
file_path = Path(r'C:/Users/AYOOLA5/Desktop/University of Toronto/MODULE 17/Module 17 Challenge Resources/LoanStats_2019Q1.csv')
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)

df.reset_index(inplace=True, drop=True)

df.head()


# # Split the Data into Training and Testing

# In[10]:


# Create a DataFrame fro
# Create our features
X = df.drop(columns="loan_status")
X = pd.get_dummies(X)
X.head()

# Create our target
y = df["loan_status"]
y


# In[11]:


X.columns


# In[12]:


X.describe()


# In[13]:


# Check the balance of our target values
y.value_counts()


# In[15]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)


# In[16]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
Scaler = StandardScaler().fit(X_train)
# Scaling the data.
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# # Ensemble Learners
# 
# In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier . For each algorithm, be sure to complete the folliowing steps:
# 
# 1. Train the model using the training data. 
# 2. Calculate the balanced accuracy score from sklearn.metrics.
# 3. Print the confusion matrix from sklearn.metrics.
# 4. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
# 5. For the Balanced Random Forest Classifier onely, print the feature importance sorted in descending order (most important feature to least important) along with the feature score
# 
# Note: Use a random state of 1 for each algorithm to ensure consistency between tests

# ### Balanced Random Forest Classifier

# In[18]:


# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

rfc = BalancedRandomForestClassifier(n_estimators=100,random_state=78).fit(X_train_scaled,y_train)
y_pred = rfc.predict(X_test_scaled)    
print(f"Training Score:{rfc.score(X_train_scaled,y_train)}")  
print(f"Training Score:{rfc.score(X_test_scaled,y_test)}") 


# In[21]:


# Calculated the balanced accuracy score
from sklearn.metrics import accuracy_score 
# Making predictions using the testing data.
predictions = rfc.predict(X_test_scaled)
predictions
acc_score = accuracy_score(y_test, predictions)
print(acc_score)


# In[22]:


# Display the confusion matrix
#from sklearn.metrics import confusion_matrix, classification_report

matrix = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    matrix, index=["Actual High-Risk", "Actual Low-Risk"], columns=["Predicted High_Risk", "Predicted Low_Risk"])
cm_df


# In[23]:


# Print the imbalanced classification report
#from imblearn.metrics import classification_report_imbalanced 
print(classification_report_imbalanced(y_test, predictions))


# In[24]:


# List the features sorted in descending order by feature importance
#from imblearn.metrics import classification_report_imbalanced 
print(classification_report_imbalanced(y_test, predictions))


# ### Easy Ensemble AdaBoost Classifier

# In[25]:


# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier

eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(X_train, y_train)


# In[26]:


# Calculated the balanced accuracy score
from sklearn.metrics import accuracy_score

acc_score2 = accuracy_score(y_test, y_pred)
print(acc_score2)


# In[27]:


# Display the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

cm_df = pd.DataFrame(
    matrix, index=["Actual High-Risk", "Actual Low-Risk"], columns=["Predicted High_Risk", "Predicted Low_Risk"])
cm_df


# In[28]:


# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred))


# In[ ]:




