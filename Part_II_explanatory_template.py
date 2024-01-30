#!/usr/bin/env python
# coding: utf-8

# # Part II - (Presentation Diabetes among people)
# ## by GHAIDA ALSAEED

# ## Investigation Overview
# 
# 
# > The goal of this presentation is to explore and communicate key insights from the 'Diabetes' dataset. We will delve into the dataset to understand factors related to diabetes patients and identify potential correlations and risk factors for diabetes. Through visualizations, we aim to convey key findings and provide a comprehensive overview of the dataset.
# 
# 
# 
# 
# ## Dataset Overview and Executive Summary
# 
# > The 'Diabetes' dataset contains information related to diabetes patients, including features such as Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome (diabetes diagnosis).
# Key Insights:
# The distribution of Glucose levels reveals variations that may indicate the prevalence and severity of diabetes in the dataset.
# BMI distribution provides insights into the weight-related characteristics of diabetes patients.
# Relationships between variables such as Glucose, Blood Pressure, Age, and BMI can reveal potential correlations and risk factors for diabetes.

# ## Key Insights:
# 
# >The distribution of Glucose levels reveals variations that may indicate the prevalence and severity of diabetes in the dataset.
# 
# >BMI distribution provides insights into the weight-related characteristics of diabetes patients.
# 
# >Relationships between variables such as Glucose, Blood Pressure, Age, and BMI can reveal potential correlations and risk factors for diabetes.

# In[3]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


# In[4]:


# load in the dataset into a pandas dataframe
df = pd.read_csv('Diabetes (2).csv')
df.head()


# ## (Visualization 1)
# 
# #  Q1: Is there a relationship between Glucose and BloodPressure among individuals with diabetes?

# In[5]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Glucose', y='BloodPressure')
plt.title('Relationship between Glucose and Blood Pressure')
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.show()


# ### The scatterplot visually demonstrates a positive correlation between Glucose levels and Blood Pressure. As Glucose levels increase, Blood Pressure tends to increase as well.

# ## (Visualization 2)
#  #  Q2: Is there a strong correlation between Glucose levels and Age?

# In[7]:


plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df, col="Outcome", height=5, aspect=1)
g.map(sns.scatterplot, "Age", "Glucose", alpha=0.5)
g.set_axis_labels("Age", "Glucose")
g.set_titles("Outcome: {col_name}")
g.add_legend()
plt.suptitle("Correlation Between Glucose and Age by Outcome", y=1.02)
plt.show()


# ### glucose levels do not appear to vary significantly with age among individuals with or without diabetes. This finding suggests that age may not be a primary factor influencing glucose levels in this dataset. Instead, other factors or variables may have a more prominent role in determining glucose levels.

# ## (Visualization 3)
# 
# # Q3 Is there a strong correlation between BMI and Age?

# In[10]:


# Set a larger figure size
plt.figure(figsize=(12, 8))

# Create the heatmap with the "coolwarm" color map
heatmap = sns.heatmap(data=df[['Age', 'BMI']].corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')

# Set the title and adjust font size
plt.title('Correlation Between Age and BMI', fontsize=16)

# Increase font size for annotations
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=12)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12)

# Add labels to axes
plt.xlabel('Age', fontsize=14)
plt.ylabel('BMI', fontsize=14)

# Ensure tight layout
plt.tight_layout()


plt.show()



# ### A correlation coefficient of 0.036 indicates a very weak positive correlation between Age and BMI. In other words, there is a slight tendency for BMI to increase as Age increases, but the relationship is not strong.
