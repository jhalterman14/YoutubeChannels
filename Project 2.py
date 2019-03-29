#!/usr/bin/env python
# coding: utf-8

# <p class="pull-left">
# ANLT 212 - Analytics Computing
# </p>
# <div class="pull-right">
# <ul class="list-inline">
# Copyright © Dana Nehoran 2017
# </ul>
# </div>

# 

# Student Names:

# Jared Halterman

# <blockquote style="border: 2px solid #666; padding: 10px; background-color: #acc;"><b>Question 1:</b> Project Definition</blockquote> 

# Select a dataset from the UCI Machine Learning Repository 
# http://archive.ics.uci.edu/ml/datasets.html
# Your dataset should be different from the one utilized on project 1. 
# 
# Describe your stakeholder. Who is the potential sponsor of your project? For example: a real estate investor, the police department, a city mayor, etc. Explain what is the objective of this project and its justification. What are the findings so far, and what is your objective.
# 
# Note: <br>
# If you want to select a dataset from a different public repository, you can.
# 

# # Data Selection: 
# ###### Explain why this data was selected. What are the overall characteristics of the data.

# The data contains Socialblade Rankings of top 5000 YouTube channels. It was selected so I could classify the different types of top Youtubers. The data contains various information on the YouTube channels such as: the Socialblade channel rankings, the grades granted by Socialblade, the YouTube channel name, the number of videos uploaded by the channel, total number of subscribers on the channel and the total number of views on all the video content by the channel.
# 
# The variables I will be using will be the numeric/continous variables: the number of videos uploaded by the channel, total number of subscribers on the channel and the total number of views on all the video content by the channel

# In[ ]:





# # Stakholder:
# ###### Describe your stakeholder. Who is the potential sponsor of your project? For example: a real estate investor, the police department, a city mayor, etc. 

# Stakeholder = Traditional Cable Companies wanting to know what the metrics are of successful Youtubers. Is it just about maxxing out your number of subscribers, or is there many paths in order to grab different niches of viewers. 

# # Objective: 
# ###### Project objective, justification, expected outcome. How your results may impact your stake holders. Which decisions or changes will the stakeholder be able to make based on this project

# My objective is to cluster the different types of successful Youtubers. Are they all similar in having just a large amounts of views, uploads and subscribers, or is there different metrics which make one a successful Youtuber. The stakeholder is looking to mimic the success the Youtubers are receiving and hope my data will help them make decisions. The company wants to create a numerous amount of channels under the same mother company and copy the upload/subscriber rate of the top Youtubers. If they can see entertainment success from someone who uploads few videos but is still successful, it may change the way they format some of their channels.

# # Background Research: 
# ###### List here all other studies related to clustering published with the same dataset, and how your proposed study is different from them.
# 
# ###### If no other clustering studies were conducted with the same dataset, you should specify: "No other studies available"

# No other studies available, that I could find. Trying to search "Youtube Channel Analysis" or "Youtube Channel Clustering", just listed Youtube videos relating to "Analysis" or "Clustering".

# <blockquote style="border: 2px solid #666; padding: 10px; background-color: #acc;"><b>Question 2:</b> Exploratory Data Analysis</blockquote> 

# #### Create some exploratory analysis on your data using core Python functions and visualizations. Cluster the information into different groups to explore the possibilities for your proposed project. Summarize your data and conduct some statistics. Explain your findings in English.

# # Data Summary: 
# ###### Show summary information of the different variables. Select the columns you are interested in. Explain each column, its range and purpose

# In[240]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from scipy.cluster.hierarchy import linkage, dendrogram
 
import pandas as pd
import numpy as np

# Set some pandas options
pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_rows', 60)
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 1000)
 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[241]:


data = pd.read_csv('youtube.csv')
df = pd.DataFrame(data)
df.head()


# Loading the data in, we get 6 columns. 
# - **Rank** - SocialBlade's ranking of the channel 
#     - Range is ascending from 1 to 5000
#     
#     
# - **Grade** - SocialBlade's grade, takes into count how views the channel is *currently* getting and other factors to determine how influential the channel is
#     - Among the channels it normalizes their influence and ranks them on a scale from A++ to F.
#     
#     
# - **Channel name** - Name of the channel
#     - Self explanatory
#     
#     
# - **Video Uploads** - Total number of videos the channel has uploaded
#     - Range is from 1 to 422,326.
#     
#     
#     
# - **Subscribers** - Total number of subscribers the channel has
#     - Max is close to 10 million. The min around 300 subscribers.
#     
#     
# - **Video views** - Total number of views among all videos the channel has
#     - Range is from 75 to 47,548,839,843
# 

# I keep the three integer columns
# 
# - **Video Uploads**
# - **Subscribers**
# - **Video views**
# 
# The purpose of these columns is to categorize the different types of successful Youtubers. For example, there may be successful Youtubers with large amounts of subscribers, yet have few videos or successful Youtubers who pump out a lot of videos and have the views to match. 

# # Data Cleaning:
# ###### Clean the data, removing rows and columns that have no useful information or no information at all

# I remove the non-integer columns and delete all the rows with "--" in their data. The "--" means they have decided not to make certain information private. I delete these columns so it removes any future errors in my script. I had some problem with the data being read as strings, so I code to make sure they are converted into numeric values.
# 
# Because of the range of numeric values, I decide to normalize the data. I also remove the outliers in order to make the normalization less skewed.

# In[242]:



df1 =df[['Video Uploads', 'Subscribers', 'Video views']]

df1.columns = ["Uploads","Subscribers","Views"]
df2=df1[~df1.Subscribers.str.contains("--")]
df3=df2[~df2.Uploads.str.contains("--")]

df = df3
df['Uploads'] = pd.to_numeric(df['Uploads'], errors='coerce')
df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')
df['Views'] = pd.to_numeric(df['Views'], errors='coerce')

df1 = df[~(df['Uploads'] > 10000)]  
df=df1
normalized_df=(df-df.mean())/df.std()
df=normalized_df

df1 = df[~(df['Views'] <= 3)]  
df=df1
normalized_df=(df-df.mean())/df.std()
df=normalized_df


df.head()


# In[ ]:





# # Adaptation: 
# ###### Create at least two additional columns that are necessary for your study. They can be calculated columns or aggregated columns.

# In[243]:


df["BigViews"] = df["Views"] > 6
df["LessThanAverageUploads"] = df["Uploads"] <0 
df.head()


# <blockquote style="border: 2px solid #666; padding: 10px; background-color: #acc;"><b>Question 3:</b> Visualization with MatPlotLib</blockquote> 

# <b><font color="blue", size = 4>a)</font> Descriptive Analytics</b>: Create at least 5 different (unique) visualizations that show different aspects of your data related to the research object of your project

# In[235]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')




ax.scatter(df.Subscribers,df.Views,df.Uploads, c='r', marker='o')

ax.set_xlabel('Subscribers')
ax.set_ylabel('Views')
ax.set_zlabel('Uploads')

plt.show()


# In[236]:


import seaborn as sns
sns.set_style('darkgrid')

sns.distplot(df.Views)


# In[238]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,5))

plt.scatter(df.Subscribers, df.Views, edgecolor = 'black', s = 80)

plt.title('Youtube Channels')
plt.xlabel('Subscribers')
plt.ylabel('Views')


# In[229]:


sns.regplot(x="Uploads", y="Views", data=df);


# In[135]:


sns.set_style('darkgrid')
sns.distplot(df.Subscribers)


# #  Outcome: 
# ###### Explain in words the outcome of your descriptive analysis

# There seems to be a very large cluster near zero, this the generic successful Youtubers who have similar ratio of subscribers/viewers/uploads. There is seemes to be a linear relationship between Subscribers and Viewers, which makes sense.

# <blockquote style="border: 2px solid #666; padding: 10px; background-color: #acc;"><b>Question 4:</b> Clustering</blockquote> 

# Select a number of numerical columns to be used for your clustering algorithms. Explain the business logic of your clustering. Select expected number of clustering to explore.

# # Cluster Analysis: 
# ###### Explain the business reason for your selection of the number of clusters.

# 4 clusters, one for each possibility.
# 
# - Low Uploads, Low Views
# 
# - Low Uploads, High Views
# 
# - High Uploads, Low Views,
# 
# - High Uploads, High Views
# 
# 
# These are the types of successful Youtubers we are trying to company in the hopes that cable companies can duplicate.

# # k-Means: 
# ###### Cluster your data using k-means. Explain your results. Try two different schemas of clusters.

# In[244]:


model = KMeans(n_clusters = 4)
model.fit(df)

fig = plt.figure(figsize=(5, 5))
 
# Create a colormap
colormap = np.array(['white', 'yellow', 'red', 'green','blue'])
 
plt.scatter(df.Uploads, df.Views, c=colormap[model.labels_], edgecolor = 'black', s = 80)
plt.title('Youtube Channels')
plt.xlabel('Uploads')
plt.ylabel('Views')


# Green - Channels with less views and uploads than most other top Youtubers, this seems to be a big group.
# 
# Red - Channels with more views and less uploads than most other top Youtubers.
# 
# Yellow - More uploads and generally more views than group Green, but not *too* many views and not *too* many uploads.
# 
# White - Channels with generally more uploads than other groups, but not all viewership is higher than other groups.

# In[289]:


model = KMeans(n_clusters = 4)
model.fit(df)

fig = plt.figure(figsize=(5, 5))
 
# Create a colormap
colormap = np.array(['white', 'yellow', 'red', 'green','blue'])
 
plt.scatter(df.Subscribers, df.Views, c=colormap[model.labels_], edgecolor = 'black', s = 80)
plt.title('Youtube Channels')
plt.xlabel('Uploads')
plt.ylabel('Subscribers')


# This is the only amount of clusters than didn't make it *too* unreadable.
# 
# - Yellow - Channels which have a generally higher amount of Uploads & Subscribers
# 
# - White - All other channels - Lower subscribers and uploads

# # Hierarchical Clustering:
# ###### Cluster your data using hierarchical clustering. Explain your results. Try two different schemas of clusters.

# <span style="background-color: #FFFF00">Your answer here</span>

# In[167]:


Z = linkage(df, 'average')
plt.figure(figsize=(25, 10))
D = dendrogram(Z=Z, orientation="right", leaf_font_size=9,)


# If we draw a horizontal line at 2, we get 9 clusters. The color scheme from the visualization show us three clusters , red, blue, and green.

# In[ ]:





# In[173]:


Z = linkage(df, 'average')
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
dendrogram(Z, leaf_font_size = 32.)
plt.show()


# In[177]:


from scipy.cluster.hierarchy import fcluster
k=3
dend_clusters = fcluster(Z, k, criterion='maxclust')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs = df.Views, ys = df.Subscribers, zs = df.Uploads, c = dend_clusters, cmap='viridis',
           s = 180, edgecolor = 'black', depthshade = True)



ax.set_xlabel('Views')
ax.set_ylabel('Subscribers')
ax.set_zlabel('Uploads')

plt.show()


# The hierarchal clustering *really* shows the number of channels that are low subscribers/low views. In the second hierarchal graph we can see how hard it skews right.

# #  kNN Clustering: 
# ###### Use your results from the cluster analysis performed in a) or b) and create a short training set. Cluster your data using your training set on kNN. Explain your results.

# In[288]:


data = pd.read_csv('datayoutube.csv')
data.values
data

df = pd.DataFrame(data)
df.columns = ["Uploads","Subscribers","Views"]
df2=df[~df.Subscribers.str.contains("--")]
df3=df2[~df2.Uploads.str.contains("--")]

df = df3
df['Uploads'] = pd.to_numeric(df['Uploads'], errors='coerce')
df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')
df['Views'] = pd.to_numeric(df['Views'], errors='coerce')
df

df1 = df[~(df['Uploads'] >= 10000)]  
df=df1
df_norm = round((df - df.min())/
                          (df.max() - df.min()),2)
df=df_norm

centroid1 = np.array([0.1,0.1])
centroid2 = np.array([0.43, 0.1])
centroid3 = np.array([.3, .3])
centroid4 = np.array([.2,1])
centroid5 = np.array([.43,.6])
centroid6 = np.array([.9,.7])


df["Dist_C1"] = np.sqrt((df.Subscribers - centroid1[0])**2 
                             + (df.Uploads - centroid1[1])**2)
df["Dist_C2"] = np.sqrt((df.Subscribers - centroid2[0])**2 + 
                             (df.Uploads - centroid2[1])**2)
df["Dist_C3"] = np.sqrt((df.Subscribers - centroid3[0])**2 + 
                             (df.Uploads - centroid3[1])**2)
df["Dist_C4"] = np.sqrt((df.Subscribers - centroid4[0])**2 + 
                             (df.Uploads - centroid4[1])**2)
df["Dist_C5"] = np.sqrt((df.Subscribers - centroid5[0])**2 + 
                             (df.Uploads - centroid5[1])**2)
df["Dist_C6"] = np.sqrt((df.Subscribers - centroid6[0])**2 + 
                             (df.Uploads - centroid6[1])**2)
df

df["Association"] = np.where(
    (df.Dist_C1 < df.Dist_C2) & 
    (df.Dist_C1 < df.Dist_C3) &
    (df.Dist_C1 < df.Dist_C4) &
    (df.Dist_C1 < df.Dist_C5) &
    (df.Dist_C1 < df.Dist_C6), 1, 
     np.where((df.Dist_C2 < df.Dist_C1) & 
    (df.Dist_C2 < df.Dist_C3) &
    (df.Dist_C2 < df.Dist_C4) &
    (df.Dist_C2 < df.Dist_C5) &
    (df.Dist_C2 < df.Dist_C6), 2,
     np.where((df.Dist_C3 < df.Dist_C1) & 
    (df.Dist_C3 < df.Dist_C2) &
    (df.Dist_C3 < df.Dist_C4) &
    (df.Dist_C3 < df.Dist_C5) &
    (df.Dist_C3 < df.Dist_C6), 3,
     np.where((df.Dist_C4 < df.Dist_C1) & 
    (df.Dist_C4 < df.Dist_C2) &
    (df.Dist_C4 < df.Dist_C3) &
    (df.Dist_C4 < df.Dist_C5) &
    (df.Dist_C4 < df.Dist_C6), 4,
     np.where((df.Dist_C5 < df.Dist_C1) & 
    (df.Dist_C5 < df.Dist_C2) &
    (df.Dist_C5 < df.Dist_C3) &
    (df.Dist_C5 < df.Dist_C4) &
    (df.Dist_C5 < df.Dist_C6), 5,6)))))
df
fig = plt.figure(figsize=(5,5))


colormap = np.array(['black', 'yellow', 'red', 'blue', 'green','pink','brown'])
 
plt.scatter(df.Subscribers,df.Uploads, 
            c = colormap[df.Association], edgecolor = 'black', s = 120)
plt.title('Types: Views / Subscribers\nLow/Low - Yellow\nMedium/Low - Red\nMedium/Medium- Blue\n High/Medium - Pink\nHigh/Low -  Green\nOddball - Brown')
ax = fig.add_subplot(111)
ax.set_xlabel('Subscribers')
ax.set_ylabel('Uploads')


# In[ ]:





# In[250]:


data = pd.read_csv('datayoutube.csv')
data.values
data

df = pd.DataFrame(data)
df.columns = ["Uploads","Subscribers","Views"]
df2=df[~df.Subscribers.str.contains("--")]
df3=df2[~df2.Uploads.str.contains("--")]

df = df3
df['Uploads'] = pd.to_numeric(df['Uploads'], errors='coerce')
df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')
df['Views'] = pd.to_numeric(df['Views'], errors='coerce')
df

df1 = df[~(df['Uploads'] >= 10000)]  
df=df1
df_norm = round((df - df.min())/
                          (df.max() - df.min()),2)
df=df_norm

centroid1 = np.array([0.1,0.1])
centroid2 = np.array([0.43, 0.1])
centroid3 = np.array([.3, .3])
centroid4 = np.array([.2,1])
centroid5 = np.array([.43,.6])
centroid6 = np.array([.9,.7])


df["Dist_C1"] = np.sqrt((df.Subscribers - centroid1[0])**2 
                             + (df.Views - centroid1[1])**2)
df["Dist_C2"] = np.sqrt((df.Subscribers - centroid2[0])**2 + 
                             (df.Views - centroid2[1])**2)
df["Dist_C3"] = np.sqrt((df.Subscribers - centroid3[0])**2 + 
                             (df.Views - centroid3[1])**2)
df["Dist_C4"] = np.sqrt((df.Subscribers - centroid4[0])**2 + 
                             (df.Views - centroid4[1])**2)
df["Dist_C5"] = np.sqrt((df.Subscribers - centroid5[0])**2 + 
                             (df.Views - centroid5[1])**2)
df["Dist_C6"] = np.sqrt((df.Subscribers - centroid6[0])**2 + 
                             (df.Views - centroid6[1])**2)
df

df["Association"] = np.where(
    (df.Dist_C1 < df.Dist_C2) & 
    (df.Dist_C1 < df.Dist_C3) &
    (df.Dist_C1 < df.Dist_C4) &
    (df.Dist_C1 < df.Dist_C5) &
    (df.Dist_C1 < df.Dist_C6), 1, 
     np.where((df.Dist_C2 < df.Dist_C1) & 
    (df.Dist_C2 < df.Dist_C3) &
    (df.Dist_C2 < df.Dist_C4) &
    (df.Dist_C2 < df.Dist_C5) &
    (df.Dist_C2 < df.Dist_C6), 2,
     np.where((df.Dist_C3 < df.Dist_C1) & 
    (df.Dist_C3 < df.Dist_C2) &
    (df.Dist_C3 < df.Dist_C4) &
    (df.Dist_C3 < df.Dist_C5) &
    (df.Dist_C3 < df.Dist_C6), 3,
     np.where((df.Dist_C4 < df.Dist_C1) & 
    (df.Dist_C4 < df.Dist_C2) &
    (df.Dist_C4 < df.Dist_C3) &
    (df.Dist_C4 < df.Dist_C5) &
    (df.Dist_C4 < df.Dist_C6), 4,
     np.where((df.Dist_C5 < df.Dist_C1) & 
    (df.Dist_C5 < df.Dist_C2) &
    (df.Dist_C5 < df.Dist_C3) &
    (df.Dist_C5 < df.Dist_C4) &
    (df.Dist_C5 < df.Dist_C6), 5,6)))))
df
fig = plt.figure(figsize=(5,5))


colormap = np.array(['black', 'yellow', 'red', 'blue', 'green','pink','brown'])
 
plt.scatter(df.Subscribers,df.Views, 
            c = colormap[df.Association], edgecolor = 'black', s = 120)
plt.title('Types: Views / Subscribers\nLow/Low - Yellow\nMedium/Low - Red\nMedium/Medium- Blue\n High/Medium - Pink\nHigh/Low -  Green\nOddball - Brown')
ax = fig.add_subplot(111)
ax.set_xlabel('Subscribers')
ax.set_ylabel('Views')


# In[ ]:





# This is the best centroid I made, it really shows the different grouping of Youtubers. I ended up using less subsets for this data set. I believe adding too many restrictions on my data, in attempts to rule out outliers, made it funky. We get a better understanding of the groups with this data.
# 
# In the yellow corner, we have the channels which have low Subscribers and low Views. The red group is more Subscribers than they have viewers. Blue is more Viewers than Subscribers. Pink is a good middle such as yellow, but a much higher volume. Green is TONS of more Viewers than they have subscribers. Lastly, brown is a channel who has a large difference of subscribers between the nearest channel and a pink amount of Viewers.
# 
# 

# <blockquote style="border: 2px solid #666; padding: 10px; background-color: #acc;"><b>Question 5:</b> Summary</blockquote> 

# # Project Summary: 
# ###### Write a few sentences about the result of this project. How can your stakeholder benefit from the results of your project? Which changes or adaptations can your stakehoders make now that he has your results?

# There is definitely different categories of successful Youtubers. My stakeholder can examine the graph and see the popularity of each type of cluster to help them determine the amount of funding/resources to spend on different type of niches. 

# # Project Report: 
# ###### Write a formal report to your stakeholders with the summary of your report 

# Greetings,
# 
# After cleaning and analyzing the data, we can see that 97% of successful Youtubers have similar ratio of subscribers and uploads. The data is in normalized to represent the relationship between each other. If we want to maximize our resources in creating and dispersing content, we should follow this trend I discovered and match the successful Youtubers exactly. I propose 97% of our channels should be "Low/Low" (as in Low Subscribers / Low Uploads) , 0.25% of our channels should be "Medium/Low", 2.2% of our channels should be "Medium/Medium", 0.4% should be "High/Medium", and 0.8% of our channels should be "High/Low" and "Oddball." 
# 
# Thank you for the funding,
# 
# Jared Halterman
# 

# In[286]:


print(df.Association.value_counts())
print(df.Association.sum())

print("Percentage of Low/Low Channels:", ((4167/4298)*100))
print("Percentage of Medium/Low Channels:",((11/4298)*100))
print("Percentage of Medium/Medium Channels:", ((95/4298)*100))
print("Percentage of High/Low Channels:", ((2/4298)*100))
print("Percentage of High/Medium Channels:", (21/4298)*100)
print("Percentage of Oddball Channels:", ((2/4298)*100))


# In[ ]:




