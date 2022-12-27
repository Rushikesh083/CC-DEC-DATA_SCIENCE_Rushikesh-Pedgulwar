#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Installing mlxtend

pip install mlxtend


# In[2]:


# Importing the libraries, Apriori and association rules


import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[3]:


#Reading Data

df = pd.read_csv("Online_Retail.csv",encoding= 'unicode_escape')
df


# In[5]:


#Data Cleanup

cart = (df[df['Country'] =="France"]
       .groupby(['InvoiceNo','Description'])['Quantity']
       .sum().unstack().reset_index().fillna(0)
       .set_index('InvoiceNo'))


# In[6]:


cart


# In[7]:


#Encoding the rows and structuring the data 

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
cart_sets = cart.applymap(encode_units)
cart_sets.drop('POSTAGE', inplace=True, axis=1)
cart_sets


# In[8]:


#Generating frequent items having Support >= 7%

frequent_items = apriori(cart_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.head()


# In[9]:


#Generating rules having (Lift >= 6) & (Confidence >= 0.8)

rules[(rules['lift'] >= 6) &
     (rules['confidence'] >= 0.8)]


# In[ ]:




