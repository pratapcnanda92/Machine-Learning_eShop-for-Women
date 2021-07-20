#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv("Data.csv", sep=";", dtype={'country':object, 'page 1 (main category)':object, 'colour':object, 'location':object, 'model photography':object, 'price 2':object})


# In[3]:


dataset.head()


# In[4]:


dataset.tail()


# In[5]:


dataset.columns


# In[6]:


dataset.info()


# In[7]:


dataset.describe()


# In[8]:


dataset[['month','day']].value_counts()


# In[9]:


dataset = dataset.drop(labels=['year','session ID'], axis=1)


# In[10]:


dataset.corr()


# In[11]:


# Categorical Tranformation of columns - Country, page 1 (main category), COLOUR, LOCATION
# Binary transformation of columns - Model Photography, price2


# In[12]:


dataset.hist(bins = 50, figsize=(20, 20))
plt.show()


# In[13]:


X = dataset.iloc[:,:].values


# In[14]:


X.shape


# In[15]:


X[1,:]


# ### Data Pre-processing - Categorical and binary variables transformations

# In[16]:


dataset.columns


# In[17]:


dataset.dtypes


# In[18]:


dataset.columns


# In[19]:


import category_encoders as ce
ce_ohe = ce.OneHotEncoder(cols = ['country', 'page 1 (main category)', 'page 2 (clothing model)', 'colour', 'location'])
d = ce_ohe.fit_transform(dataset)
d.head()


# In[20]:


X = d.iloc[:,:].values
X.shape


# ### Using the elbow method to find the optimal number of clusters

# In[21]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[22]:


wcss


# ### Training the K-Means model on the dataset

# In[23]:


# from above plot of WCSS, we can see that the cluster is 5

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[24]:


y_kmeans


# In[25]:


y_final = np.reshape(y_kmeans, (len(y_kmeans),1))
y_final


# In[26]:


newdata_array = np.concatenate((X, y_final), axis=1)
newdata_array


# In[27]:


# code for getting columns for our new dataset
np.concatenate((d.columns, np.array(['y_final'])),axis=0)


# In[28]:


newset = pd.DataFrame(newdata_array, columns=np.concatenate((d.columns, np.array(['y_final'])),axis=0))
newset.head()


# ### Regression Models

# In[29]:


X


# In[30]:


y_kmeans


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_kmeans, test_size = 0.2, random_state = 42)


# In[32]:


X_test.shape


# In[33]:


X_train.shape


# In[34]:


# Importing required libraries of Scikit Learn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[35]:


PipelineLR = Pipeline([
    ('scalar', StandardScaler()),
    ('LinearReg', LinearRegression())
])

PipelineDT = Pipeline([
    ('scalar', StandardScaler()),
    ('DT', DecisionTreeRegressor(random_state = 0))
])

PipelineRF = Pipeline([
    ('scalar', StandardScaler()),
    ('RandomForest', RandomForestRegressor(n_estimators=10, random_state=0))
])


# In[36]:


PipelineLR.fit(X_train, y_train)


# In[37]:


PipelineDT.fit(X_train, y_train)


# In[38]:


PipelineRF.fit(X_train, y_train)


# In[39]:


y_LR = np.round(PipelineLR.predict(X_test),decimals=0)


# In[40]:


y_DT = np.round(PipelineDT.predict(X_test), decimals=0)


# In[41]:


y_RF = np.round(PipelineRF.predict(X_test), decimals=0)


# In[42]:


print(f"Regression Scores \nLinear Regression: {np.round(r2_score(y_LR, y_test),decimals=4)} \nDecision Tree: {np.round(r2_score(y_DT, y_test),decimals=4)} \nRandom Forest: {np.round(r2_score(y_LR, y_test),decimals=4)}")

