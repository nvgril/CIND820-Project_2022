#!/usr/bin/env python
# coding: utf-8

# CIND 820 Big Data Analytics Project
# 
# Name: Nelly Grillo
# 
# Student number: 501144764
# 
# Supervisor: Ceni Babaoglu, Ph.D
# 

# Big Data Analytics Project

# # Data Description

# Gender: Gender of the passengers (Female, Male)
# 
# Customer Type: The customer type (Loyal customer, disloyal customer)
# 
# Age: The actual age of the passengers
# 
# Type of Travel: Purpose of the flight of the passengers (Personal travel, Business travel)
# 
# Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
# 
# Flight distance: The flight distance of this journey
# 
# Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not applicable; 1-5)
# 
# Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient
# 
# Ease of Online booking: Satisfaction level of online booking
# 
# Gate location: Satisfaction level of Gate location
# 
# Food and drink: Satisfaction level of Food and drink  
# 
# Online boarding: Satisfaction level of Online boarding
# 
# Seat comfort: Satisfaction level of Seat comfort
# 
# Inflight entertainment: Satisfaction level of Inflight entertainment
# 
# On-board service: Satisfaction level of On-board service
# 
# Leg room service: Satisfaction level of Leg room service
# 
# Baggage handling: Satisfaction level of Baggage handling
# 
# Check-in service: Satisfaction level of Check-in service
# 
# Inflight service: Satisfaction level of Inflight service
# 
# Cleanliness: Satisfaction level of Cleanliness
# 
# Departure Delay in Minutes: Minutes delayed when departure
# 
# Arrival Delay in Minutes: Minutes delayed when Arrival
# 
# Satisfaction: Airline Satisfaction level (Satisfied, neutral or dissatisfied)

# # Library Imports

# In[96]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[97]:


# Loading the dataset
df_train = pd.read_csv('train.csv', index_col='id')
df_test = pd.read_csv('test.csv')


# In[98]:


df_train.columns


# In[99]:


df_train.shape


# * 'Unnamed:0' will be deleted

# In[100]:


# Drop "Unnamed" 
df_train = df_train.drop('Unnamed: 0', axis=1)
df_train = df_train.sort_values('id', ascending= True)

df_test = df_test.drop('Unnamed: 0', axis=1)
df_test = df_test.sort_values('id', ascending= True)


# In[101]:


# Looking at first few instances
df_train.head()


# In[102]:


print("The data shape is: {}".format(df_train.shape))


# In[103]:


df_train.info()


# In[104]:


df_train.nunique()[:10].sort_values(ascending=False)


# # Data Cleaning
# 
# 1. NaN Values

# In[105]:


df_train.isnull().sum()


# There are 310 missing values in the Arrival Delay in Minutes row. To avoid skewing the data, the NaN values will be dropped.

# In[106]:


# Dropping NaN rows
df_train = df_train.dropna().copy()

print("The data shape is: {}".format(df_train.shape))


# 2. Duplicate Values

# In[107]:


df_train.duplicated().any()


# # Descriptive Statistics

# In[108]:


df_train.describe().T


# 3. Outliers

# In[109]:


# Looking for outliers
numer_features = df_train.select_dtypes(exclude=['object'])
numer_features.columns


# In[110]:


fig = plt.figure(figsize=(12,18))
for i in range(len(numer_features.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=numer_features.iloc[:,i])

plt.tight_layout()
plt.show()


# Here and from the descriptives statistics table, we can see that there are large values for the Departure Delay in Minutes and Arrival Delay in Minutes: 1592 and 1584 respectively.

# In[111]:


sns.boxplot(x=df_train['Departure Delay in Minutes'])


# In[112]:


sns.boxplot(x=df_train['Arrival Delay in Minutes'])


# The boxplots above shows that the two values (1592 and 1584) are significantly larger than the rest of values. Both will be removed.

# In[113]:


plt.scatter(df_train['Departure Delay in Minutes'], np.random.rand(df_train.shape[0]))
plt.scatter(df_train['Arrival Delay in Minutes'], np.random.rand(df_train.shape[0]))


# In[114]:


df_train.loc[df_train['Departure Delay in Minutes'] > 1200]
df_train.loc[df_train['Arrival Delay in Minutes'] > 1200]


# In[115]:


print("The data shape is: {}".format(df_train.shape))


# In[116]:


outliers = df_train[df_train['Arrival Delay in Minutes'] > 1250].index
df_train.drop(outliers, inplace=True)
print("The data shape is: {}".format(df_train.shape))


# There are 23 columns of data and some of them are categorical. 

# In[117]:


# Categorical data


# In[118]:


categ_columns = df_train.select_dtypes(include = ['object'])
unique_values = categ_columns.nunique(dropna = False)
print(unique_values)


# There are 5 categorical columns: Gender, Customer Type, Type of Travel and Satisfaction contains 2 possible values, and Class contains 3 possible values.

# In[ ]:




