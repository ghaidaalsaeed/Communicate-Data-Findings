#!/usr/bin/env python
# coding: utf-8

# # Part I - (Exploraty)
# ## by (GHAIDA ALSAEED)
# 
# ## Introduction

# 
# In this report, we will explore the 'Diabetes' dataset, which contains information related to diabetes patients. We aim to understand the dataset's structure, distributions, and relationships between variables.
#   

# ## Preliminary Wrangling

# In[2]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Diabetes (2).csv')

# Describe dataset properties
df.info()


# ### What is the structure of your dataset?
# 
# > The dataset contains information about diabetes patients, including various features related to their health and medical history.
# 
# ### What is/are the main feature(s) of interest in your dataset?
# 
# > The main features of interest in this dataset include Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome.
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# > To support my investigation into diabetes-related features, I will consider the provided columns: Glucose, BloodPressure, Insulin, BMI, and Age.
# 
# 

# ## Univariate Exploration
# 
# > In this section, investigate distributions of individual variables. If you see unusual points or outliers, take a deeper look to clean things up and prepare yourself to look at relationships between variables.
# 

# ### Question 1: What is the distribution of Glucose levels?
# ### Let's create a histogram to visualize the distribution of Glucose levels.

# In[4]:


plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Glucose', bins=20, kde=True)
plt.title('Distribution of Glucose Levels')
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
plt.show()


# ### The histogram displays a peak in the distribution of Glucose levels around the 100-110 range. This suggests that a significant number of individuals in the dataset have Glucose levels in this range.
# 
# ### The KDE curve provides a smooth estimate of the probability density function. It confirms the bimodal nature of the distribution and highlights the two peaks.
# 

# ### Question 2: What is the distribution of BMI (Body Mass Index)?
# ### Let's create a countplot to visualize the distribution of BMI.

# In[5]:


# Create a violin plot for BMI
plt.figure(figsize=(10, 6))
sns.violinplot(x='BMI', data=df, color='skyblue')
plt.title('Distribution of BMI (Body Mass Index)')
plt.xlabel('BMI')
plt.ylabel('Density')

# Show the plot
plt.show()


# ### The violin plot shows that the highest density of BMI values is concentrated in the range of approximately 30 to 40. This range represents the most common BMI values in the dataset.
# ### The violin plot's width on both sides of the central peak indicates a relatively symmetric distribution of BMI values. This means that there is a substantial number of individuals with both lower and higher BMI values.
# 

# ## Bivariate Exploration
# 
# > Relationship between Glucose and Blood Pressure
# Let's create a scatter plot to visualize the relationship between Glucose levels and Blood Pressure.

# In[17]:


plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Glucose', y='BloodPressure')
plt.title('Relationship between Glucose and Blood Pressure')
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.show()


# ### The scatterplot visually demonstrates a positive correlation between Glucose levels and Blood Pressure. As Glucose levels increase, Blood Pressure tends to increase as well.

# Relationship between Age and BMI
# Let's create a heatmap and a boxplot to visualize the relationship between Age and BMI.

# In[18]:


plt.figure(figsize=(10, 6))
sns.heatmap(data=df[['Age', 'BMI']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Relationship Between Age and BMI')
plt.tight_layout()

# Show the chart
plt.show()


# ### A correlation coefficient of 0.036 indicates a very weak positive correlation between Age and BMI. In other words, there is a slight tendency for BMI to increase as Age increases, but the relationship is not strong.

# In[7]:


# Define age categories
age_bins = [0, 25, 60, float("inf")]  # Define age ranges: 0-25, 26-60, 61 and older
age_labels = ['Young (0-25)', 'Mid (26-60)', 'Senior (61+)']  # Define labels for the age categories

# Create a new column 'AgeCategory' based on the defined bins and labels
df['AgeCategory'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Create the boxplot with AgeCategory on the x-axis
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='AgeCategory', y='BMI', palette='Set2')
plt.title('Boxplot: Relationship Between Age and BMI')
plt.xlabel('Age Category')
plt.ylabel('BMI')

# Set custom x-axis labels
plt.xticks(ticks=[0, 1, 2], labels=age_labels)

plt.show()


# ### ndividuals in the 'Mid (26-60)' age group tend to have higher BMI values compared to those in the 'Young (0-25)' and 'Senior (61+)' age groups. This suggests that middle-aged individuals may be at a higher risk of having elevated BMI, which can be associated with various health implications, such as obesity-related health issues.

# # Multivariate

# Relationship between Glucose, BMI, and Outcome
# Let's create a scatter plot matrix to explore the relationships between Glucose, BMI, and the Outcome (diabetes diagnosis).
# 
# 

# In[20]:


sns.pairplot(data=df[['Glucose', 'BMI', 'Outcome']], hue='Outcome', palette='Set1')
plt.suptitle('Relationships between Glucose, BMI, and Diabetes Outcome')
plt.show()


# ###  The pairplot distinguishes data points based on the 'Outcome' variable (diabetes diagnosis). It shows that individuals diagnosed with diabetes ('Outcome' = 1, indicated by orange points) generally have higher Glucose levels and BMI compared to those without diabetes ('Outcome' = 0, indicated by blue points). This aligns with the clinical understanding that diabetes is associated with higher Glucose levels and may also involve weight-related factors.

# Is there a strong correlation between Glucose levels and Age?

# In[22]:


plt.figure(figsize=(12, 8))
g = sns.FacetGrid(df, col="Outcome", height=5, aspect=1)
g.map(sns.scatterplot, "Age", "Glucose", alpha=0.5)
g.set_axis_labels("Age", "Glucose")
g.set_titles("Outcome: {col_name}")
g.add_legend()
plt.suptitle("Correlation Between Glucose and Age by Outcome", y=1.02)
plt.show()


# ### glucose levels do not appear to vary significantly with age among individuals with or without diabetes. This finding suggests that age may not be a primary factor influencing glucose levels in this dataset. Instead, other factors or variables may have a more prominent role in determining glucose levels.

# ## Conclusions
# >In this exploration of the 'Diabetes' dataset, we analyzed the distribution of Glucose levels, the distribution of BMI, relationships between Glucose and Blood Pressure, and relationships between Age and BMI. Some key findings include:
# 
# The distribution of Glucose levels may indicate the prevalence and severity of diabetes in the dataset.
# BMI distribution provides insights into the weight-related characteristics of diabetes patients.
# Relationships between variables such as Glucose, Blood Pressure, Age, and BMI can reveal potential correlations and risk factors for diabetes.

# In[ ]:





# In[ ]:




