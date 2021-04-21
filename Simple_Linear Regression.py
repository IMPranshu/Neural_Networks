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
df.corr()
del df['parks']
import statsmodels.api as sn
x= sn.add_constant(df['room_num'])
lm=sn.OLS(df['price'],x).fit()
from sklearn.linear_model import LinearRegression
y = df['price']
x = df[['room_num']]
lm2=LinearRegression()
lm2.fit(x,y)
sns.jointplot(x =df['room_num'],y=df['price'],data =df,kind = 'reg')


# In[ ]:




