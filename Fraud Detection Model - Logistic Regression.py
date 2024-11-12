#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
import numpy as np 


# loading the csv file with data on credit card transactions
df = pd.read_csv('credit_card_Fraud_log.csv')
print(df.head(10))


# In[6]:


# data types of the columns in the dataframe
print(df.info())


# In[7]:


# summary statistics of amount
print(df.amount.describe())


# In[8]:


# creating a column of payments
df['IsPayment'] = df['type'].apply(lambda x: 1 if x in (['PAYMENT', 'DEBIT']) else 0)


# In[9]:


# creating a column of cash movement transactions
df['IsMovement'] = df['type'].apply(lambda x: 1 if x in (['CASH_OUT', 'TRANSFER']) else 0)


# In[10]:


# creating a column of the absolute differences between the origin & destination account 
df['accountDiff'] = abs(df.oldbalanceDest - df.oldbalanceDest)


# In[11]:


# looking at the df with the added columns
# print(df.head(5))
print(df.info())


# In[12]:


# creating label and features variables for our model 
label = df['isFraud']
# Converting features into an array ensures consistent data types. 
# This is important since mixed data types (e.g., integers and floats) can lead to errors or performance issues in some algorithms.
features = np.array(df[['amount', 'IsPayment', 'IsMovement', 'accountDiff']]) 


# In[13]:


#splitting the data into train & test sets, with test set= 0.3

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)


# In[15]:


# normalize the features variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[21]:


# fitting the logistic regression Model 
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)



# In[19]:


# scoring the model on training data 
print(model.score(X_train, y_train))


# In[20]:


# scoring the model on test data 
print(model.score(X_test, y_test))


# In[22]:


#printing model coefficients to determine the most important feature
print(model.coef_)

# IsMovement seems (cash-outs or transfers) seem to be most important predictor of fradulent transcations


# In[29]:


# predicting new transactions using the model 
# sample transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
transaction4 = np.array([400037.05, 1, 1, 290000.05])

#combining the transactions into a single array
sample_transactions = np.stack([transaction1, transaction2, transaction3, transaction4])



# In[30]:


#normalizing the features of sample transctions
sample_transactions = scaler.transform(sample_transactions)
#predicting the fradulent transactions from the sample data
print(model.predict(sample_transactions))

#none of the transactions is fradulent


# In[28]:


#to see the probabilities that lead to the above predictions
print(model.predict_proba(sample_transactions))

#column 1  is the probability of a transaction not being fraudulent, 
#a 2nd column is the probability of a transaction being fraudulent


# In[ ]:




