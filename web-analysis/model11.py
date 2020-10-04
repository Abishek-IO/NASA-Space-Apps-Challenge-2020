#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import plotly
import plotly.offline as py
import plotly.graph_objs as go
import datetime as dt
import plotly.express as px
import matplotlib.pyplot as plot
py.init_notebook_mode()
import pickle


# In[3]:


temp_data = pd.read_csv("archive.csv")
temp_data.head(11)


# In[52]:


from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[94]:


x_learn = temp_data.iloc[:,[0,2]].values
y_learn = temp_data["Mean"]


# In[95]:


xTrain, xTest, yTrain, yTest = train_test_split(x_learn, y_learn, test_size = 1/3, random_state = 0)


# In[96]:


linearRegressor = LinearRegression()


# In[97]:


linearRegressor.fit(xTrain, yTrain)


# In[111]:



value = [[CO2_value,year]]
#value=value.reshape(-1,1)
yPrediction = linearRegressor.predict(value)


pickle.dump(linearRegressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))
print(yPrediction)

