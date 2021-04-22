#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sns
df=pd.read_csv('C:/Users/Specter/Desktop/resources/Data Files/2. ST Academy - Classification models resource files/House-Price.csv',header =0)


# In[9]:


X = df[['price']]


# In[10]:


y = df[['Sold']]


# In[11]:


X.head()


# In[12]:


y.head()


# In[13]:


#Using Sklearn

from sklearn.linear_model import LogisticRegression


# In[14]:


clf_lrs = LogisticRegression()


# In[15]:


clf_lrs.fit( X, y)


# In[16]:


clf_lrs.coef_


# In[17]:


clf_lrs.intercept_


# In[18]:


#2nd Method using stats model
import statsmodels.api as sn


# In[19]:


X_cons = sn.add_constant(X)


# In[20]:


X_cons.head()


# In[21]:


import statsmodels.discrete.discrete_model as sm


# In[23]:


logit = sm.Logit(y, X_cons).fit()


# In[24]:


logit.summary()


# In[ ]:




