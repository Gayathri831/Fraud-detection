#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_excel("C:/Users/gayat/Documents/Fraud.xlsx")


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.rename(columns={'newbalanceOrig':'newbalanceOrg'}, inplace = True)


# In[6]:


data.head()


# In[7]:


data.drop(columns = ['nameOrig','nameDest'],axis=1, inplace=True)


# In[8]:


data.head()


# In[9]:


print('minimum value of an amount, the new/old of an Origin and Destination')

data[['amount','oldbalanceOrg','newbalanceOrg','oldbalanceDest','newbalanceDest']].min()


# In[10]:


print('miximum value of an amount, the new/old of an Origin and Destination')

data[['amount','oldbalanceOrg','newbalanceOrg','oldbalanceDest','newbalanceDest']].max()


# # Data Analysis

# In[11]:


var = data.groupby('type').amount.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
var.plot(kind='bar')
ax1.set_title("Total amount per transaction type")
ax1.set_xlabel('Type of Transaction')
ax1.set_ylabel('Amount');


# In[12]:


data.loc[data.isFraud == 1].type.unique()


# In[13]:


sns.heatmap(data.corr(),cmap='RdBu');


# In[14]:


fraud = data.loc[data.isFraud == 1]
nonfraud = data.loc[data.isFraud == 0]


# In[15]:


fraudcount = fraud.isFraud.count()
nonfraudcount = nonfraud.isFraud.count()


# In[16]:


sns.heatmap(fraud.corr(),cmap='RdBu',);


# In[17]:


print('The total number of fraud transaction is {}.'.format(data.isFraud.sum()))
print('The total number of fraud transaction which is marked as fraud {}.'.format(data.isFlaggedFraud.sum()))
print('Ratio of fraud transaction vs non-fraud transaction is 1:{}.'.format(int(nonfraudcount//fraudcount)))


# In[18]:


print('Thus in every 773 transaction there is 1 fraud transaction happening.')
print('Amount lost due to these fraud transaction is ${}.'.format(int(fraud.amount.sum())))


# In[19]:


piedata = fraud.groupby(['isFlaggedFraud']).sum()


# In[20]:


f, axes = plt.subplots(1,1, figsize=(6,6))
axes.set_title("% of fraud transaction detected")
piedata.plot(kind='pie',y='isFraud',ax=axes, fontsize=14,shadow=False,autopct='%1.1f%%');
axes.set_ylabel('');
plt.legend(loc='upper left',labels=['Not Detected','Detected'])
plt.show()


# In[21]:


fig = plt.figure()
axes = fig.add_subplot(1,1,1)
axes.set_title("Fraud transaction which are Flagged Correctly")
axes.scatter(nonfraud['amount'],nonfraud['isFlaggedFraud'],c='g')
axes.scatter(fraud['amount'],fraud['isFlaggedFraud'],c='r')
plt.legend(loc='upper right',labels=['Not Flagged','Flagged'])
plt.show()


# # Data Exploration

# In[22]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(nonfraud['oldbalanceOrg'],nonfraud['amount'],c='g')
ax.scatter(fraud['oldbalanceOrg'],fraud['amount'],c='r')
plt.show()


# In[23]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['oldbalanceDest'])
plt.show()


# In[24]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['newbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceDest'])
plt.show()


# In[25]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['newbalanceDest'])
ax.scatter(fraud['step'],fraud['oldbalanceDest'])
plt.show()


# In[26]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceOrg'])
plt.show()


# # Data cleaning

# In[27]:


import pickle


# In[28]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[30]:


data = data.replace(to_replace={'PAYMENT':1,'TRANSFER':2,'CASH_OUT':3,
                                            'CASH_IN':4,'DEBIT':5,'No':0,'Yes':1})


# In[31]:


data


# # machine learning model

# In[33]:


X = data.drop(['isFraud'],axis=1)
y = data[['isFraud']]


# In[43]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 121)


# In[45]:


#NORMALIZATION STANDARD SCALER
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[51]:


from sklearn.linear_model import LogisticRegression
algorithm = LogisticRegression()
algorithm.fit(X_train, y_train)


# In[53]:


#predicting and testing
y_predict = algorithm.predict(X_test)
y_predict


# In[54]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score

cm = confusion_matrix(y_test,y_predict)
cm


# In[55]:


ac = accuracy_score(y_test,y_predict)
ac


# In[ ]:




