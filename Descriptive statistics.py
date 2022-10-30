#!/usr/bin/env python
# coding: utf-8

# In[18]:


#loading the dataset
import pandas as pd
df_train = pd.read_csv('train.csv', index_col='id')
df_test = pd.read_csv('test.csv')


# In[20]:


df_train


# In[21]:


df_train.head()


# In[23]:


# Drop "Unnamed" row
df_train = df_train.drop('Unnamed: 0', axis=1)
df_train = df_train.sort_values('id', ascending= True)


# In[24]:


df_train.head()


# In[27]:


print("The data shape is: {}".format(df_train.shape))


# In[28]:


df_train.info()


# In[30]:


df_train.nunique()[:10].sort_values(ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


df_train.describe()


# 

# In[ ]:





# In[ ]:





# In[ ]:




