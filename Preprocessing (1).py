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


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
df=pd.read_csv('C:/Users/Specter/Desktop/resources/Data Files/1. ST Academy - Crash course and Regression files/House_Price.csv',header =0)
uv=np.percentile(df.n_hot_rooms,[99])[0]


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[ ]:


import pandas as pd


# In[ ]:


import numpy as np
import matplotlib as mpl
print('spam')


# In[ ]:


# This is an example snippet!
# To create your own, add a new snippet block to the
# snippets.json file in your jupyter nbextensions directory:
# /nbextensions/snippets/snippets.json
import this


# In[1]:


# This is an example snippet!
# To create your own, add a new snippet block to the
# snippets.json file in your jupyter nbextensions directory:
# /nbextensions/snippets/snippets.json
import this


# In[1]:


pwd


# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[4]:


df.head()


# In[5]:


get_ipython().system('jt -l')


# In[15]:


df.head()


# In[16]:


df.shape


# In[17]:


df.describe()


# In[18]:


sns.jointplot(x='n_hot_rooms',y='price',data=df) 


# In[20]:


sns.jointplot(x='rainfall',y='price',data=df)


# In[21]:


df.head()


# In[22]:


sns.countplot(x='airport',data=df)


# In[ ]:





# In[23]:


sns.countplot(x='waterbody',data=df)


# In[24]:


sns.countplot(x='bus_ter',data=df)


# In[ ]:


##Observations
#1.Missing values in n_hos_beds.
#2.Skewness or outliers in crime  rate.
#3.Outliers in n_hot_rooms and rainfall.
#4.Bus_ter has only one value


# In[25]:


df.info()


# In[26]:


np.percentile(df.n_hot_rooms,[99])


# In[56]:


np.percentile(df.n_hot_rooms,[99])[0]


# In[3]:


uv=np.percentile(df.n_hot_rooms,[99])[0]


# In[4]:


df[(df.n_hot_rooms > uv)]


# In[6]:


df.n_hot_rooms[(df.n_hot_rooms > 3*uv)]= 3*uv


# In[6]:


df[(df.n_hot_rooms > uv)]


# In[7]:


lv=np.percentile(df.rainfall,[1])[0]


# In[8]:


lv


# In[9]:


df[(df.rainfall<lv)]


# In[11]:


df.rainfall[(df.rainfall < .3*lv )]= .3*lv


# In[12]:


df.head()


# In[37]:


df[(df.rainfall<uvr)]


# In[38]:


df[(df.rainfall > .3*)]


# In[40]:


df.head()


# In[41]:


df.head()


# In[2]:


df=pd.read_csv('C:/Users/Specter/Desktop/resources/Data Files/1. ST Academy - Crash course and Regression files/House_Price.csv',header =0)


# In[61]:


df[(df.rainfall < lv)]


# In[63]:


sns.jointplot(x='crime_rate',y='price',data=df)


# In[64]:


df.describe()


# In[13]:


df.info()


# In[14]:


#impute missing values

df.n_hos_beds=df.n_hos_beds.fillna(df.n_hos_beds.mean())


# In[15]:


df.info()


# In[6]:


sns.jointplot(x='crime_rate',y='price',data=df)


# In[16]:


df.crime_rate= np.log(1+df.crime_rate)


# In[8]:


sns.jointplot(x='crime_rate',y='price',data=df)


# In[9]:


np.help()


# In[17]:


df.crime_rate= np.log(1+df.crime_rate)
sns.jointplot(x='crime_rate',y='price',data=df)


# In[17]:


df.crime_rate= np.tanh(df.crime_rate)


# In[18]:


sns.jointplot(x='crime_rate',y='price',data=df)


# In[18]:


df['avg_dist'] = (df.dist1+df.dist2+df.dist3+df.dist4)/4


# In[19]:


df.describe()


# In[21]:


df.dist1(max)


# In[25]:


df[(df.dist1[max])]


# In[26]:


np.max(df.dist1)


# In[27]:


max_dist1=np.max(df.dist1)
max_dist2=np.max(df.dist2)
max_dist3=np.max(df.dist3)
max_dist4=np.max(df.dist4)
min_dist1=np.min(df.dist1)
min_dist2=np.min(df.dist2)
min_dist3=np.min(df.dist3)
min_dist4=np.min(df.dist4)


# In[28]:


df['max_dist_avg']=(max_dist1+max_dist2+max_dist3+max_dist4)/4
df['min_dist_avg']=(min_dist1+min_dist2+min_dist3+min_dist4)/4


# In[29]:


df.describe()


# In[30]:


(max_dist1+max_dist2+max_dist3+max_dist4)/4


# In[31]:


(min_dist1+min_dist2+min_dist3+min_dist4)/4


# In[32]:


np.max(df.dist1)


# In[33]:


np.max(df.dist1)


# In[34]:


np.max(df.dist1,df.dist2)


# In[43]:


arr=(df.dist1,df.dist2,df.dist3,df.dist4)
df['max_dist_avg'] = np.max([(df.dist1,df.dist2,df.dist3,df.dist4)])


# In[38]:


df['min_dist_avg'] = np.min(arr)


# In[44]:


df.describe()


# In[40]:


arr=(df.dist1,df.dist2,df.dist3,df.dist4)


# In[41]:


arr=(df.dist1,df.dist2,df.dist3,df.dist4)
arr


# In[42]:


np.max(arr)


# In[20]:


del df['dist1']


# In[19]:


df.describe()


# In[21]:


del df['dist2']
del df['dist3']
del df['dist4']


# In[50]:


df.describe()


# In[22]:


del df['bus_ter']


# In[23]:


df=pd.get_dummies(df)


# In[24]:


df.describe()


# In[22]:


df.head()


# In[25]:


del df['airport_NO']
del df['waterbody_None']


# In[24]:


df.head()


# In[26]:


df.corr()


# In[27]:


del df['parks']


# In[28]:


from sklearn.datasets import load_iris


# In[29]:


iris = load_iris()


# In[30]:


iris


# In[30]:


imprt statsmodel.api as sn


# In[31]:


import statsmodels.api as sn
x= sn.add_constant(df['room_num'])


# In[32]:


lm=sn.OLS(df['price'],x).fit()


# In[33]:


lm.summary()


# In[34]:


from sklearn.linear_model import LinearRegression


# In[35]:


y = df['price']


# In[37]:


x = df[['room_num']]
lm2=LinearRegression()
lm2.fit(x,y)


# In[38]:


print(lm2.intercept_, lm2.coef_)


# In[40]:


sns.jointplot(x =df['room_num'],y=df['price'],data =df,kind = 'reg')


# In[42]:


sns.jointplot(x =df['poor_prop'],y=df['price'],data =df,kind = 'reg')


# In[ ]:




