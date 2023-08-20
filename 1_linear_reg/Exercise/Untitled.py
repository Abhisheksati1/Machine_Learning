#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[10]:


df = pd.read_csv("/home/abhishek/Documents/canada_per_capita_income.csv")
df


# In[7]:


feg = linear_model.LinearRegression()


# In[9]:


feg.fit(df[['year']],df['per capita income (US$)'])


# In[12]:


feg.predict([[2020]])


# In[13]:


feg.predict([[2030]])


# In[ ]:




