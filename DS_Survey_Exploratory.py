#!/usr/bin/env python
# coding: utf-8

# In[215]:


import pandas as pd
pd.set_option('display.max_columns', 50)  
pd.set_option('display.max_rows', 50)  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('autosave', '120')


# In[216]:


survey = pd.read_csv("kaggle_survey_2021_responses.csv")
survey.shape


# In[217]:


survey.head()


# # Introduction
# 

# This is an exploratory analysis of the 2021 Kaggle Data Science Survey. This was a survey done on the Kaggle website of those in the data science industry. An invitation to the survey was sent to all Kaggle members in late 2021. The survey can be found at https://www.kaggle.com/c/kaggle-survey-2021. This dataset contains 25,973 observations and 38 variables (questions). The purpose of this analysis is to do an exploratory analysis of the survey since I am looking to join the data science industry. In particular, I am interested in the answers to the questions about education, programming languages, industries, and BI tools. 
# 
# The first thing I did was a bit of data cleaning. Since I am going to be looking for a job in the US, I filtered the dataset to include only the observations where the country was the United States. I am also going to be looking for a job with the title of "data analyst" or "data scientist" most likely, so I filtered the results again to only include these job titles. The resulting dataframe only contained 699 observations. This subset is considerably smaller than the original dataset, but still large enough to get some useful information from an exploratory analysis. 

# In[218]:


# Deleting first row 
survey_mod = survey.iloc[1:,:]
survey_mod.head()


# In[6]:


survey_mod.shape


# In[219]:


# Pulling out country of USA and jobs of Data Scientist/Data Analyst only
survey_mod.Q3.value_counts()


# In[220]:


survey_mod.Q5.value_counts()


# In[221]:


survey_mod2 = survey_mod[(survey_mod['Q3'] == "United States of America")] 
survey_mod2.shape


# In[222]:


survey_mod2 = survey_mod2[(survey_mod.Q5 == "Data Scientist") | (survey_mod.Q5 == "Data Analyst")] 
survey_mod2.shape

# Alternative method
#filter1 = survey_mod["Q3"].isin(["United States of America"])
#filter2 = survey_mod["Q5"].isin(["Data Scientist", "Data Analyst"])
#survey_mod[filter1 & filter2].shape


# In[223]:


survey_mod2


# In[12]:


# Getting % of null values for each question
survey_mod2.isnull().sum()/survey_mod.shape[0]


# # Education
# 

# The first question I wanted to analyze was about education. I was curious as to what is the educational background of data analysts and scientists in the US. In particular, I was wondering if they generally had a master's degree, a bachelor's degree, or perhaps just some professional certificates. 

# In[224]:


survey_mod2["Q4"].value_counts()


# In[225]:


# Rename to "Some college" for graph
survey_mod2.loc[(survey_mod2.Q4 == 'Some college/university study without earning a bachelor’s degree'),'Q4'] = 'Some college'
survey_mod2["Q4"].value_counts()


# In[226]:


# Filter out all degrees not in the top 4 for better visual
survey_mod_edplt = survey_mod2[(survey_mod['Q4'] != "Professional doctorate") & (survey_mod['Q4'] != "I prefer not to answer") 
& (survey_mod['Q4'] != "No formal education past high school")]
survey_mod_edplt["Q4"].value_counts()


# In[227]:


# Creating df of value_counts
survey_mod_edplt2 = survey_mod_edplt["Q4"].value_counts(normalize=True).to_frame()
survey_mod_edplt2["Q4"] = 100 * survey_mod_edplt2["Q4"]
survey_mod_edplt2


# In[242]:


# Exploring Education(4)
plt.figure(figsize=(10,6))
plt.title ("Highest Education Attained by Data Scientists and Data Analysts in the US", fontsize = 13.5)
sb.barplot(x = survey_mod_edplt2.index, y = survey_mod_edplt2['Q4'])
plt.ylabel("Percentage", fontsize = 12)
plt.show()


# As this bar graph shows, the majority of data scientists/analysts had a Master's degree (57%) and over three-fourths had a graduate degree of some kind (master's or doctorate). Only 3% of respondents hadn't graduated college. It appears as if a Bachelor's degree is the minimum requirement to get a data scientist/analyst position and a graduate degree is the best way to go. It would have been interesting to know what the majors of the respondents were, but this question wasn't asked. 

# # Programming Languages

# The next question I wanted to explore was programming languages. I wanted to see which programming languages were most often used on a regular basis by data analysts and data scientists in the US. This was question seven in the survey. The next question asked which programming language would the respondent recommend a prospective data scientist or analyst to learn first. I wanted to explore this question as well. 

# In[243]:


# Exploring Programming Languages (7/8)
# survey_mod2 is base dataset
survey_mod2.head()


# In[244]:


# Pulling out question 7
Q7 = survey_mod2.loc[:,survey_mod2.columns.str.startswith('Q7')]
Q7.head()


# In[245]:


# Changing column names
Q7.columns = list(Q7.mode().values)
Q7.head()


# In[247]:


# Getting percentages of each language
Q7_sums = Q7.count().reset_index()
Q7_sums[0] = Q7_sums[0]/699 * 100
Q7_sums


# In[248]:


# Taking 'other' out
Q7_sums2 = Q7_sums.iloc[0:12,:]
Q7_sums2


# In[249]:


# Renaming columns before making bargraph
Q7_sums2.rename(columns = {'level_0': 'Language', 0: 'Percentage'}, inplace=True)
Q7_sums2


# In[259]:


# Graph highlighting the top 3
plt.figure(figsize=(10,6))
plt.title ("Programming Languages Used by Data Scientists and Analysts in the US", fontsize = 13.5)
bar_clr = ['grey' if (x < 30) else 'mediumblue' for x in Q7_sums2['Percentage']] # Highlighting top 3
Q7_graph = sb.barplot(x = "Language", y = "Percentage", data = Q7_sums2, palette=bar_clr,
             order = Q7_sums2.sort_values("Percentage", ascending = False).Language)
Q7_graph.set_xlabel("Language", fontsize = 12)
Q7_graph.set_ylabel("Percentage", fontsize = 12)
plt.show()


# In[258]:


# Pulling out Q8 and summing responses
Q8_sums = survey_mod2['Q8'].value_counts(normalize=True).to_frame()
Q8_sums.rename(columns = {"Q8": "Percentage"}, inplace = True)
Q8_sums["Percentage"] = Q8_sums["Percentage"] * 100
Q8_sums


# In[262]:


# Graph of question 8
plt.figure(figsize=(10,6))
plt.title ("Programming Languages Recommended for Newbies", fontsize = 13.5)
bar_clr = ['grey' if (x < 30) else 'mediumblue' for x in Q8_sums['Percentage']] # Highlighting top 3
Q8_graph = sb.barplot(x = Q8_sums.index, y = "Percentage", data = Q8_sums, palette=bar_clr)
plt.xlabel("Language", fontsize = 12)
Q8_graph.set_ylabel("Percentage", fontsize = 12)
plt.show()


# There are three programming lanaguages that data scientists and analysts in this survey use on a regular basis: *python*, *SQL*, and *R*. Python and SQL are used on a regular basis by the majority of respondents, while R is used on a regular basis by nearly half of the respondents. However, thre is one clear winner when it comes to which one is recommended for aspiring data scientists/analysts to learn first. Nearly 70% of respondents in the survey answered Python to this question, while SQL and R each only represented 10% of the answers. The takeaway here seems to be that someone who wants to get into the data science/analyst field needs to be skilled in at least one of the top 3 programming languages, and the most important one according to the respondents to this survey is python. 

# ## Data Visualization Tools

# The next question I wanted to explore was about data visualization tools. This question asked what data visualization tools were used on a regular basis by those who answered the survey. This question was similar to question 7 in that respondents could give multiple answers to the question. I expected the top 3 visualization tools to be *seaborn*, *matplotlib*, and *ggplot* but didn't know what order they would be in. 

# In[263]:


# Exploring Data Vis Tools (14)
Q14 = survey_mod2.loc[:,survey_mod2.columns.str.startswith('Q14')]
# Changing names of columns to the correct tool
Q14.columns = list(Q14.mode().values)
Q14.head()


# In[264]:


# Getting percentages of each visualization tool
Q14_sums = Q14.count().reset_index()
Q14_sums[0] = Q14_sums[0]/699 * 100
Q14_sums


# In[265]:


# Taking 'other' out
Q14_sums2 = Q14_sums.iloc[0:11,:]
Q14_sums2


# In[266]:


# Renaming columns 
Q14_sums2.rename(columns = {'level_0': 'Vis_Tool', 0: 'Percentage'}, inplace=True)
Q14_sums2


# In[272]:


# Graph of vis tools
plt.figure(figsize=(10,6))
plt.title ("Visualization Tools Regularly Used", fontsize = 13.5)
bar_clr = ['grey' if (x < 30) else 'darkgreen' for x in Q14_sums2['Percentage']] # Highlighting top 3
vis_graph = sb.barplot(x = Q14_sums2.Vis_Tool, y = "Percentage", data = Q14_sums2, palette=bar_clr,
           order = Q14_sums2.sort_values("Percentage", ascending = False).Vis_Tool)
vis_graph.set_xticklabels(['Matplotlib', 'Seaborn', 'Ggplot', 'Plotly', 'Shiny', 'Bokeh', 'None', 'Leaflet',
                    'D3 js', 'Geoplotlib', 'Altair']) # Manually shortening names
vis_graph.set_ylabel("Percentage", fontsize = 12)
plt.xlabel("Tool", fontsize = 12)
plt.show()


# The four main visualization tools used by data scientists/analysts in this survey were *matplotlib*, *seaborn*, *ggplot*, and *plotly*. Matplotlib and seaborn were used by the majority of respondents, while ggplot and plotly were used by about 40% of the respondents. All other visualization tools were used by less than 20% of those who answered the survey. So it appears that an aspiring data analyst or scientist needs to learn at least one of these top four to be career ready. 

# ## Industry and Company Size

# The next two questions I wanted to focus on were type of industry and company size of the data scientists/analysts that answered this survey. I expected technology companies and perhaps insurance companies to be near the top in the industry category and I also thought that most data scientists/analysts probably worked for large companies. 

# In[275]:


# Pulling out Q20 and summing responses
Q20_sums = survey_mod2['Q20'].value_counts(normalize=True).to_frame()
Q20_sums.rename(columns = {"Q20": "Percentage"}, inplace = True)
Q20_sums["Percentage"] = Q20_sums["Percentage"] * 100
Q20_sums


# In[274]:


# Only getting top 8 to include in bar graph
Q20_sums_8 = Q20_sums.iloc[0:8]
Q20_sums_8


# In[280]:


# Graph of Industries
plt.figure(figsize=(10,6))
plt.title ("Top Industries of Data Scientists/Analysts in the US", fontsize = 13.5)
ind_graph = sb.barplot(x = Q20_sums_8.index, y = "Percentage", data = Q20_sums_8)
ind_graph.set_xticklabels(['Technology', 'Finance', 'Health', 'Education', 'Other', 'Government', 'Insurance',
                           'Retail'])
ind_graph.set_ylabel("Percentage", fontsize = 12)
plt.xlabel("Industry", fontsize = 12)
plt.show()


# In[281]:


# Company Size
Q21_sums = survey_mod2['Q21'].value_counts(normalize=True).to_frame()
Q21_sums.rename(columns = {"Q21": "Percentage"}, inplace = True)
Q21_sums["Percentage"] = Q21_sums["Percentage"] * 100
Q21_sums


# In[282]:


# Changing x axis order from largest to smallest
order = ['10,000 or more employees', '1000-9,999 employees', '250-999 employees', 
         '50-249 employees', '0-49 employees']


# In[294]:


# Graph of company size
plt.figure(figsize=(12,8))
plt.title ("Company Size of Data Scientists/Analysts in the US", fontsize = 14)
Q21_graph = sb.barplot(x = Q21_sums.index, y = "Percentage", data = Q21_sums, order = order) # Changing the order
Q21_graph.set_ylabel("Percentage", fontsize = 12.5)
plt.xlabel("Number of Employees", fontsize = 12.5)
plt.show()


# This analysis shows that data scientists and analysts works in a variety of industries. Unsurprisingly, technology is the most common industry, but other industries such as healthcare, finance, and education are not that far behind. This was mostly what I expected, except that I was a bit surprised that education was one of the top sectors that data scientists/analysts worked in. The bar graph above also shows that these jobs are found in companies of every size. Nearly one-third of repsondents worked in huge companies of at least 10,000 or more employees and over half of respondents worked in companies with 1,000 or more employees. However, nearly 20% worked in very small companies with fewer than 50 employees and almost half (46%) worked in small to medium sized companies who had fewer than 1,000 employees.  

# ## BI Tools

# The last question I wanted to explore was which BI tools were often used by those who currently work as data scientists or analysts. Currently, I am learning Tableau, and I was curious whether Tableau or Power BI is used more often by those who currently have an analytics job. In addition, I wanted to know how common it was for other BI Tools, besides Tableau or Power BI, to be used on a regular basis. 

# In[286]:


# Exploring BI Tools (35)
Q35_sums = survey_mod2['Q35'].value_counts(normalize=True).to_frame()
Q35_sums.rename(columns = {"Q35": "Percentage"}, inplace = True)
Q35_sums["Percentage"] = Q35_sums["Percentage"] * 100
Q35_sums


# In[287]:


# Making a new df manually/If there were more categories need to manipulate this data at beginning
data = [['Tableau', 45.93], ['Power BI', 21.48], ['Other', 32.59]]
Q35_sums2 = pd.DataFrame(data, columns = ['Tool', 'Percentage'])
Q35_sums2      


# In[293]:


# Donut Chart of BI Tools
plt.figure(figsize=(10,6))
circle = plt.Circle( (0,0), 0.7, color='white') # creating hole in middle
explode = (0.10, 0.05, 0.05) # separation in slices
colors = ['royalblue', 'indianred', 'goldenrod']
plt.pie(Q35_sums2['Percentage'], 
        labels=Q35_sums2['Tool'],
        autopct='%1.1f%%', pctdistance=0.85,
        explode=explode,
        colors = colors,
        startangle = 270)
# Adding circle to pie chart
fig = plt.gcf()
fig.gca().add_artist(circle)
  
plt.title('BI Tools 2021')
plt.show()


# Tableau was the most common BI tool used in this survey. Nearly half of respondents answered that this was the BI tool they used the most often. This was more than twice the amount that said Power BI was the BI tool they used most regularly. This was a bit surprising since my impression was that Power BI had been gaining on Tableau as the go-to BI tool for analytics. But it seems that, at least in this survey, Tableau is the clear favorite among data scientists/analysts in the US. I am also a bit surprised that nearly a third of respondents said that neither Tableau nor Power BI was their BI tool of choice. To be fair, the tool with the third highest number of responses (Google Data Studio) only received 7.4% of the responses, which is far lower than Tableau (45.9%) and Power BI (21.5%). Yet it is significant that all other BI tools combined were favored by nearly one third of data scientists/analysts in this survey. 

# ## Conclusion

# This analysis explored the Kaggle Data Science Survey that was conducted in 2021. Since I am interested in getting a data analyst/scientist position in the US, I filtered the survey data to only include those that were in the US who currently had one of these job titles. The data indicates that education is important for getting a job with the minimum required education being a bachelor's degree. However, a graduate degree would seem to open up more opportunites as a majority of those in the survey held either a master's degree or Phd. The survey also suggested that learning one of the three main programming languages (Python, R, SQL) would be beneficial with most respondents saying that Python was the most important of the three. Finally, learning either Tableau or Power BI would seem to be a good idea to be job ready. 
# 
# The data was not so clear about what types of industries a prospective data scientists/analyst should be looking to get into. The technology sector is an obvious choice, but healthcare, finance, and even education seem to have a large number of analytics positions as well. Similarly, there seems to be no clear answer about whether most data scientists/analysts work in large companies or small companies. Nearly half of respondents said they worked for a company with 1,000 or more employees, while nearly half stated that their company had fewer than 1,000 employees. 
# 
# This is an interesting data set, especially for someone who is looking to get into the analytics field, but there are a couple of big caveats to be aware of. First, this data set may *not* reflect the attitudes of all data scientists/analysts in the US. This was a survey sent specifically to Kaggle users, so the best we could say is that it represents the views of Kaggle users who are currently data scientists/analysts in the US. However, this claim turns out to be dubious because according to the documentation the survey was sent to all Kaggle users and everyone that filled out the survey was included in the data. This means that this is a *voluntary response sample* and not a random sample. Therefore, we cannot make any confident claims about either the opinions of data scientists/analysts in the US or about Kaggle users who hold those positions in the US based on this survey alone. 
# 
# 
