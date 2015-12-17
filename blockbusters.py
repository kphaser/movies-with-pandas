
# coding: utf-8

# # Exploring Blockbuster Movies With Pandas

# I love movies so I thought it would be fun to analyze some movie data and show off what pandas can do. In this notebook I explore a crowdsourced dataset of Blockbuster movies from 1975 to 2014. The dataset includes the top 10 most popular movies of each year based on reviews/ratings from IMDB and Rotten Tomatoes as well as ticket sales information.

# In[1]:

# Standard import
get_ipython().magic(u'matplotlib inline')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series


# The two important features of pandas are the Data Frame and Series. You can see that I imported them above. A data frame is essentially a table. A series is a single column of our data frame.

# ### Data Loading and Preprocessing

# Read in the csv file using pd.read_csv('file_path'). You can see that I added a parameter to parse the release date column using parse_dates = ['column_name']

# In[2]:

blockbusters = pd.read_csv('top_ten_movies_per_year_DFE.csv', parse_dates=['release_date'])


# Now that we've imported our data. We can run .info() to give us a general view of what we have. I like to do this so I can get a general overview of what I have.

# In[3]:

blockbusters.info()


# So .info() tells us that our object blockbusters is a data frame and it lists each column and the number of valid rows and the data types of those columns. It looks like we have a total of 399 rows and 20 columns. Something to note is that Genre_2 and Genre_3 column only have 369 and 269 valid rows which likely means that we have missing data. More about that later.

# Let's take a look at a small snippet of our imported data frame to see what we have. We can get the first 5 rows of our data frame using .head() method.

# In[4]:

blockbusters.head()


# Similarly, we can see the last few rows of the dataset using .tail()

# In[5]:

blockbusters.tail()


# So it looks like two columns, adjusted and worldwide_gross are currencies. Pandas recognizes these columns as strings. Let's also get rid of the $ and comma in adjusted and worldwide_gross columns and convert those columns to floats so we run statistics on those columns later.

# In[6]:

blockbusters['adjusted'] = blockbusters['adjusted'].replace('[\$,]','',regex=True).astype(float)

blockbusters['worldwide_gross'] = blockbusters['worldwide_gross'].replace('[\$,]','',regex=True).astype(float)


# All I did here was replace the dollar signs and commas with nothing and convert the values of the column to float.

# It looks pretty good, but not quite. For our particular purposes, the poster_url column is not useful. So let's get rid of that column. It also makes our data frame more compact more easy to look at by getting rid of that column.

# In[7]:

blockbusters = blockbusters.drop('poster_url',1)

blockbusters.head()


# Notice I specified the column and also put 1 as a parameter. The structure of a pandas data frame is that 0 refers to the row axis and 1 refers to the column axis. Because I was deleting a column, I had to specify 1 otherwise it would give me an error because it defaults to looking at the row axis.

# We can get a little more information by running .describe() on our data to get some general descriptive statistics that may be useful to us.

# In[8]:

blockbusters.describe()


# Notice that running .describe() did not display all the columns. This is because that method expects column values to be numeric in nature so anything that isn't numeric won't be displayed in the output. (Note: Earlier I mentioned 'adjusted' and 'worldwide_gross' would be important columns to look at which is why I cleaned those columns up earlier otherwise running .describe() would not produce anything).
# 
# You can make some general observations from the descriptive statistics produced. Some that I thought were interesting are the average imdb rating of blockbuster movies is 7.05 and average length of films is 119.2 minutes.

# ### Selecting Data

# We can select just specific columns. Note: I run .head() on most of my queries below to prevent huge outputs.

# In[9]:

blockbusters['title'].head()


# Pandas gives us an alternative way of doing this is by calling .column_name after the data frame. The result is the same as the previous.

# In[10]:

blockbusters.title.head()


# We can also select multiple columns.

# In[11]:

blockbusters[['title','imdb_rating']].head()


# Notice the double brackets. This allows us to select multiple columns by passing a list into our selection and return only those parts of the dataframe back.
# 
# We can also do slicing, which is just selecting a range of rows.

# In[12]:

blockbusters[0:3]


# We can use the .value_counts() on our data frame to get a count of what we want. I find this method to be particularly useful for many cases as you'll see below. Let's see how many movies are there for each year in our data.

# In[13]:

blockbusters['year'].value_counts().sort_index()


# Notice I had to add .sort_index() after .value_counts() because if I didn't then python likely would give me the results in random order.
# 
# So right now our index has no meaning and goes from 0 to the length of the dataset. It's a generic index given by python. However, we can set our index to be something more meaningful like using title.

# In[14]:

blockbusters.set_index('title').sort_index().head()


# Now our dataset is indexed by the title right? Not exactly. Keep in mind that if you call the blockbusters data frame, it'll return the original dataset without the release date as index. What that means is that pandas does not mutate your data frame, but it returns a new data frame when you do something to it. So if you want to keep changes you made to a data frame, remember to assign it to a variable.

# In[15]:

title = blockbusters.set_index('title').sort_index()


# If you have a meaningful index, you can use .ix() to select your data frame by index name.

# In[16]:

title.ix['Guardians of the Galaxy']


# Let's answer some more interesting questions below regarding our data.

# ### What are the top 10 highest rated sci-fi films?

# In[17]:

blockbusters[(blockbusters.Genre_1 == 'Sci-Fi') | (blockbusters.Genre_2 == 'Sci-Fi') | (blockbusters.Genre_3 == 'Sci-Fi')][['title','imdb_rating','Genre_1','Genre_2','Genre_3']].sort_values(by='imdb_rating',ascending=False).head(10)


# ### How many movies do we have for each MPAA rating group?

# In[18]:

ratings = blockbusters['rating'].value_counts()
ratings


# It looks like our dataset contains mostly PG-13 rated movies. Let's plot this below.

# In[19]:

ratings.plot(kind='bar')
plt.title("MPAA Ratings Frequencies")
plt.xlabel("Rating")
plt.ylabel("Frequency")


# Below I will start using the groupby function quite a bit. It's an extremely useful method that will allow us to group our data frame so we can ask more interesting questions.

# ### Which years had the highest average rating?

# In[20]:

blockbusters.groupby('year').mean()[['imdb_rating']].sort_values('imdb_rating',ascending=False).head()


# ### Which genre of movie has the highest gross?

# In[21]:

blockbusters.groupby('Genre_1').sum()[['worldwide_gross']].sort_values('worldwide_gross',ascending=False).head()


# ### Which studios earned the highest average amount per movie?

# In[22]:

blockbusters.groupby('studio').mean()[['worldwide_gross']].sort_values('worldwide_gross',ascending=False).head()


# ### Which studios released the most films?

# In[23]:

blockbusters.studio.value_counts().head()


# That's it for now! There's obviously still a lot to pandas I didn't cover here. I only went over the basics in terms of reading, understanding, adjusting and selecting your data. There are much more complex capabilities such as joining datasets, hierarchical indexing, and much more. Stayed tuned for more pandas goodness!

# Dataset retrieved from: http://www.crowdflower.com/data-for-everyone
