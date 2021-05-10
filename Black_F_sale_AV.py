#!/usr/bin/env python
# coding: utf-8

# In[50]:


# import libraries

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[51]:


plt.figure(figsize = (20, 17))


# In[52]:


train=pd.read_csv("train.csv")
test= pd.read_csv("test.csv")


# In[53]:


train.head()


# In[54]:


test.head()


# In[55]:


train.info()


# In[56]:


train.describe().transpose()


# In[57]:


train.describe(include="object")


# In[83]:


train["Age"].unique()


# In[84]:


train["City_Category"].unique()


# In[85]:


train["Stay_In_Current_City_Years"].unique()


# In[58]:


sns.pairplot(train)
plt.show()


# In[59]:


sns.heatmap(train.corr(),annot=True)
plt.show()


# In[60]:


train.isnull().sum().sort_values()


# In[61]:


train['source'] = 'train'
test['source'] = 'test'


# In[62]:


saledata = pd.concat([train, test])


# In[63]:


saledata.info()


# In[64]:


saledata['Stay_In_Current_City_Years'] = dataset['Stay_In_Current_City_Years'].apply(lambda x : str(x).replace('4+', '4'))


# In[65]:


saledata['Age'] = dataset['Age'].apply(lambda x : str(x).replace('55+', '55'))


# In[66]:


saledata.head(10)


# In[67]:


saledata.drop('User_ID', axis = 1, inplace = True)


# In[68]:


saledata.drop('Product_Category_3', axis = 1, inplace = True)


# In[69]:


saledata.drop('Product_ID', axis = 1, inplace = True)


# In[70]:


saledata.info()


# In[71]:


saledata['Product_Category_2'].fillna(dataset['Product_Category_2'].median(), inplace = True)


# In[72]:


saledata.info()


# In[73]:


from sklearn.preprocessing import LabelEncoder


# In[74]:


le=LabelEncoder()


# In[76]:


saledata['Gender']=le.fit_transform(saledata['Gender'])


# In[82]:


saledata.info()


# In[78]:


saledata['Age']=le.fit_transform(saledata['Age'])


# In[80]:


saledata['City_Category']=le.fit_transform(saledata['City_Category'])


# In[86]:


saledata['Stay_In_Current_City_Years']=saledata['Stay_In_Current_City_Years'].astype('int')


# In[87]:


saledata.info()


# In[91]:


train_f=saledata.loc[saledata['source']=='train']
test_f=saledata.loc[saledata['source']=='test']


# In[93]:


train_f.drop('source',axis=1,inplace=True)
test_f.drop('source',axis=1,inplace=True)


# In[103]:


X = train_f.iloc[:,train_f.columns!="Purchase"]


# In[104]:


Y = train_f["Purchase"]


# In[136]:


from sklearn.model_selection import train_test_split
# X_train,Y_train,X_test,Y_test= train_test_split(X,Y,test_size=.3,random_state=10)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.3,random_state=10)


# In[137]:


X_train


# In[138]:


Y_train


# In[139]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# In[140]:


dt=DecisionTreeRegressor()


# In[141]:


dt.fit(X_train,Y_train)


# In[142]:


test_f=test_f.iloc[:,test_f.columns!="Purchase"]


# In[143]:


test


# In[144]:


test_f


# In[149]:


prd=dt.predict(X_test)


# In[150]:


prd


# In[152]:


rmse=np.sqrt(mean_squared_error(Y_test,prd))


# rmse

# In[153]:


rmse


# In[154]:


purchase=dt.predict(test_f)


# In[155]:


purchase


# In[157]:


pd.DataFrame(purchase).to_csv("C:\\Users\\BKY\\AIML\\Analytics Vidya\\Black_F_sale\\purchae.csv")


# In[ ]:




