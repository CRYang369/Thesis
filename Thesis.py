#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


from collections import OrderedDict


# In[5]:


import seaborn as sns


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


arrhyData=pd.read_csv('arrhythmia data with title.csv')


# In[8]:


arrhyData.head()


# In[9]:


arrhyData.describe()


# In[10]:


arrhyData.describe().T


# In[11]:


fig=plt.figure(figsize=(16,5))


# In[12]:


df1=arrhyData[arrhyData["Sex"]==1]


# In[13]:


df1.shape


# In[14]:


df2=arrhyData[arrhyData["Sex"]==0]


# In[15]:


df2.shape


# In[16]:


arrhyData.info()


# In[17]:


arrhyData.tail()


# In[18]:


arrhyData.groupby("label").describe()


# In[19]:


arrhyData.groupby("Sex").describe()


# In[26]:


sns.countplot(x='label',data=arrhyData)


# In[21]:



sns.boxplot(x='label',y='Sex',data=arrhyData)


# In[24]:


sns.boxplot(x='label',y='Age',data=arrhyData)


# In[27]:


sns.boxplot(x='label',y='Height',data=arrhyData)


# In[28]:


sns.boxplot(x='label',y='Weight',data=arrhyData)


# In[49]:


arrhyData.label>1


# In[50]:


arrhyData.label==1


# In[57]:


type=pd.value_counts(arrhyData['label']==1)


# In[69]:


sns.countplot(x=type,data=arrhyData)


# In[45]:


sns.boxplot(x=range(normal arrhythmia),data=arrhyData )


# In[22]:


if('label'==1):
    kind='normal'
else:
    kind='arrythmia'


# In[23]:


sns.countplot(x='kind',data=arrhyData)


# In[26]:


#相关系数矩阵
rDf=examDf.corr()
rDf


# In[28]:


#将训练集特征转化为二维数组**行1列
X_train=X_train.reshape(-1,1)
#将测试集特征转化为二维数组**行1列
X_test=X_test.reshape(-1,1)

