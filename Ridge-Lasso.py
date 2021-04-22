#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_multi,y_multi, test_size = .2, random_state = 0)
lm_a = LinearRegression()
lm_a.fit(X_train,y_train)
y_test_a = lm_a.predict(X_test) #predicted
y_train_a = lm_a.predict(X_train)
from sklearn.metrics import r2_score
r2_score(y_test, y_test_a) #Use this to evaluate the performance of the model


# In[2]:


#We need to standardise the data
from sklearn import preprocessing


# In[3]:


#we create the scalar object whihc will store the scaling info of X variable
scalar = preprocessing.StandardScaler().fit(X_train)


# In[4]:


X_train_s = scalar.transform(X_train)


# In[5]:


X_test_s = scalar.transform(X_test)


# In[6]:


#Ridge Regression

from sklearn.linear_model import Ridge


# In[7]:


lm_r = Ridge(alpha = 0.5)
lm_r.fit(X_train_s, y_train)


# In[8]:


r2_score(y_test, lm_r.predict(X_test_s))


# In[9]:


#Validaion Curve
from sklearn.model_selection import validation_curve


# In[10]:


#We will check for different values of alpha
param_range = np.logspace(-2,8,100)
param_range#values of our alpla


# In[30]:


train_scores, test_scores = validation_curve(Ridge(), X_train_s, y_train, 'alpha', param_range, scoring= 'r2' )


# In[13]:


print(train_scores)


# In[14]:


print(test_scores)


# In[17]:


train_mean = np.mean(train_scores, axis =1)


# In[18]:


test_mean = np.mean(test_scores, axis =1)


# In[19]:


max(test_mean)


# In[20]:


sns.jointplot(x=np.log(param_range), y = test_mean)


# In[46]:


np.where(test_mean==max(test_mean))


# In[48]:


param_range[30]


# In[23]:


lm_r_best = Ridge(alpha=param_range[30])


# In[27]:


lm_r_best.fit(X_train_s,y_train)


# In[28]:


r2_score(y_test, lm_r_best.predict(X_test_s))


# In[29]:


r2_score(y_train, lm_r_best.predict(X_train_s))


# In[31]:


#Lasso Regression

from sklearn.linear_model import Lasso
lm_l=Lasso(alpha = 0.4)


# In[32]:


lm_l.fit(X_train_s,y_train)


# In[33]:


r2_score(y_test,lm_l.predict(X_test_s))


# In[34]:


train_scores, test_scores = validation_curve(Lasso(),X_train_s, y_train, 'alpha', param_range , scoring='r2')


# In[40]:


train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
np.where(test_mean==max(test_mean))


# In[41]:


lm_l_best = Lasso(alpha=param_range[7])


# In[43]:


lm_l_best.fit(X_train_s, y_train)
r2_score(y_test, lm_l_best.predict(X_test_s))


# In[44]:


r2_score(y_train, lm_l_best.predict(X_train_s))


# In[ ]:




