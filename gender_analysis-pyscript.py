#!/usr/bin/env python
# coding: utf-8

# # Gender Analysis
# Include information and graphs that help answering the following questions
# * Do men or women dominate speaking roles in Hollywood movies?
# * Has gender balance in speaking roles changed over time?
# * Do films in which men do more speaking make a lot more money than films in which women speak more?

# ## Some graphs to include
# Plot the following by year
# * General proportion of male lead.
# * Proportion of words spoken.
# * Lead age average by gender
# * Gross vs gender speaking percentage

import pandas as pd
import numpy as np
import seaborn as sns
from  matplotlib.ticker import PercentFormatter
sns.set_theme(style="darkgrid")

# Load the dataset
url = 'data/train.csv'
data = pd.read_csv(url)

# Dropping 0-word entries

# Get indexes for which the number of words of female or male is 0
dropIndex = data[ (data['Number words female'] == 0) | (data['Number words male'] == 0) ].index

# Delete the rows from the dataset
data.drop(dropIndex , inplace=True)
data['Number words co-lead'] = data['Number of words lead'] - data['Difference in words lead and co-lead']


# Index by year for easier management of data
data.reset_index(inplace=True)
data.set_index('Year', inplace=True) 
data.sort_index(ascending=True, inplace=True);

# Select years to focus on (start in 1981, before there is little data)
years = np.array(data.index.unique())
years = years[years > 1980]

years_dict = {'Year': years} # create dict of years for feeding a pandas dataframe
analysis_df = pd.DataFrame(data=years_dict) # create a dataframe with a first column 'Year'
analysis_df.set_index('Year');


# Populate analysis_df with relevant stats, consolidated by year

m_lead_p_year = []
f_lead_p_year = []
m_words_p_year = []
f_words_p_year = []
f_age_mean = []
m_age_mean = []
f_lead_age_mean = []
m_lead_age_mean = []
m_actors_p = []
f_actors_p = []


for year in years:
    
    year_df = data.loc[[year]] # Create a DF with the year (making sure it comes as a matrix)
    
    # % of male lead actors
    num_movies = len(year_df)
    m_lead_p_year.append((year_df['Lead'] == 'Male').sum() / num_movies)
    f_lead_p_year.append((year_df['Lead'] == 'Female').sum() / num_movies)
    
    # % of words spoken by male and female
    num_words_male = year_df['Number words male'].sum()
    num_words_male += year_df[(year_df['Lead'] == 'Male')]['Number of words lead'].sum()
    m_words_p_year.append(num_words_male / year_df['Total words'].sum())
    
    num_words_female = year_df['Number words female'].sum()
    num_words_female += year_df[(year_df['Lead'] == 'Female')]['Number of words lead'].sum()
    f_words_p_year.append(num_words_female / year_df['Total words'].sum())
    
    # Female and male age average
    f_age_mean.append(year_df['Mean Age Female'].mean())
    m_age_mean.append(year_df['Mean Age Male'].mean())
    
    # Female and male lead age average
    f_lead_age_mean.append(year_df[(year_df['Lead'] == 'Female')]['Age Lead'].mean())
    m_lead_age_mean.append(year_df[(year_df['Lead'] == 'Male')]['Age Lead'].mean())
    
    
analysis_df['Male lead %'] = m_lead_p_year
analysis_df['Female lead %'] = f_lead_p_year
analysis_df['Male words %'] = m_words_p_year
analysis_df['Female words %'] = f_words_p_year
analysis_df['Mean age female'] = f_age_mean
analysis_df['Mean age male'] = m_age_mean
analysis_df['Female lead age mean'] = f_lead_age_mean
analysis_df['Male lead age mean'] = m_lead_age_mean


analysis_df.describe()

analysis_df.tail()

male_lead_plot = sns.relplot(data=analysis_df, x=analysis_df['Year'], y='Male lead %', kind='line');
male_lead_plot.set(xlim=(1980,2015), ylim=(0,1), title="Male dominance of leading roles")
male_lead_plot.ax.axline(xy1=(1980, 0.5), slope=0, color="r", dashes=(5, 2))
male_lead_plot.ax.yaxis.set_major_formatter(PercentFormatter(1))


male_words_plot = sns.relplot(data=analysis_df, x=analysis_df['Year'], y='Male words %', kind='line');
male_words_plot.set(xlim=(1980,2015), ylim=(0,1), title="Male dominance of words spoken")
male_words_plot.ax.axline(xy1=(1980, 0.5), slope=0, color="r", dashes=(5, 2))
male_words_plot.ax.yaxis.set_major_formatter(PercentFormatter(1))

male_lead_plot = sns.relplot(data=analysis_df, x=analysis_df['Year'], y='Mean number of male actors', kind='line');
male_lead_plot.set(xlim=(1982,2015))


male_lead_plot = sns.relplot(data=analysis_df, x=analysis_df['Year'], y='Mean number of female actors', kind='line');
male_lead_plot.set(xlim=(1982,2015))


# ## Other general statistics

m_words_p = [] # Male words %
f_words_p = [] # Female words %

meanw_per_actor = [] # Mean words per actor
meanw_per_actress = [] # Mean words per actress

for (idx, row) in data.iterrows():
    total_words = row.loc['Total words']
    words_male = row.loc['Number words male']
    words_female = row.loc['Number words female']
    words_lead = row.loc['Number of words lead']
    num_actress = row.loc['Number of female actors']
    num_actors = row.loc['Number of male actors']
    
    lead = row.loc['Lead']
    if lead == 'Male':
        words_male += words_lead
        m_words_p.append(words_male / total_words)
        f_words_p.append(words_female / total_words)
        meanw_per_actor.append(m_t_words / num_actors)
        meanw_per_actress.append(words_female)
    else:
        words_female += words_lead
        f_words_p.append(words_female / total_words)
        m_words_p.append(words_male / total_words)
        meanw_per_actor.append(words_male / num_actors)
        meanw_per_actress.append(words_female / num_actress)
        
    
data['Male words %'] = m_words_p
data['Female words %'] = f_words_p
data['Mean words per actress'] = meanw_per_actress
data['Mean words per actor'] = meanw_per_actor


data.head()

data.describe()

gross_plot = sns.relplot(data=data, x=data['Gross'], y=data['Male words %']);
gross_plot.ax.axline(xy1=(200, 0.5), slope=0, color="r", dashes=(5, 2))
gross_plot.ax.axline(xy1=(200, 0.70), slope=0, color="g", dashes=(4, 10))
gross_plot.ax.yaxis.set_major_formatter(PercentFormatter(1))


# Above the red dotted line are the movies where male actors speak more than 50% of the words (gender balance).
# 
# Above the green dotted line are the movies where male actors speak more than 75% of the words (overall male speaking % mean).
# 
# On both accounts, movies with higher gross (X axis) tend to have more % of the words spoken by males.

# ## Other tests
# Below are other tests.. for example, exploring if there are analysis respecting age. For example, the mean age of females in films is 7 years younger than males. Can other interesting relationships can be seen?


female_age_average_plot = sns.relplot(data=analysis_df, x=analysis_df['Year'], y='Mean age female', kind='line');
female_age_average_plot.set(ylim=(0,60));


female_age_average_plot = sns.relplot(data=analysis_df, x=analysis_df['Year'], y='Mean age male', kind='line');
female_age_average_plot.set(ylim=(0,60));

