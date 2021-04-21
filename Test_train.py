#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
df=pd.read_csv('C:/Users/Specter/Desktop/resources/Data Files/1. ST Academy - Crash course and Regression files/House_Price.csv',header =0)
uv=np.percentile(df.n_hot_rooms,[99])[0]
lv=np.percentile(df.rainfall,[1])[0]
df.rainfall[(df.rainfall < .3*lv )]= .3*lv
df.n_hos_beds=df.n_hos_beds.fillna(df.n_hos_beds.mean())
df.crime_rate= np.log(1+df.crime_rate)
df['avg_dist'] = (df.dist1+df.dist2+df.dist3+df.dist4)/4
del df['dist1']
del df['dist2']
del df['dist3']
del df['dist4']
del df['bus_ter']
df=pd.get_dummies(df)
del df['airport_NO']
del df['waterbody_None']
del df['parks']
import statsmodels.api as sn
from sklearn.linear_model import LinearRegression
X_multi = df.drop('price',axis = 1)
y_multi = df['price']
X_multi_cons = sn.add_constant(X_multi)
lm_multi = sn.OLS(y_multi, X_multi_cons).fit()
lm3 = LinearRegression()
lm3.fit(X_multi,y_multi)
print(lm3.intercept_, lm3.coef_)


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X_multi,y_multi, test_size = .2, random_state = 0)


# In[5]:


print(X_train.shape, X_test.shape,y_train.shape,y_test.shape)


# In[6]:


lm_a = LinearRegression()


# In[7]:


lm_a.fit(X_train,y_train)


# In[8]:


y_test_a = lm_a.predict(X_test) #predicted


# In[9]:


y_train_a = lm_a.predict(X_train)


# In[11]:


from sklearn.metrics import r2_score


# In[12]:


r2_score(y_test, y_test_a) #Use this to evaluate the performance of the model


# In[13]:


r2_score(y_train,y_train_a)


# In[ ]:




