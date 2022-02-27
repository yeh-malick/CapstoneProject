#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Setting up librabies & programs used throughout the process ie matplotlib, pandas, numpy, opendatasets, wordcloud, regular expressions, textblob
get_ipython().system('pip install opendatasets')


# In[5]:


import opendatasets as od


# In[14]:


dataset ='https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets/version/113'


# In[7]:


od.download(dataset)


# In[15]:


import os
data_dir = 'all-covid19-vaccines-tweets'


# In[16]:


os.listdir(data_dir)
import pandas as pd
tweets_df = pd.read_csv('vaccination_all_tweets.csv')


# In[54]:


tweets1 = tweets.loc[:,['id', 'user_location','date','text']]


# In[25]:


tweets1['user_location'].unique()


# In[47]:


tweets1_data_list = ['United States']


# In[51]:


tweets = tweets1[tweets1.user_location.isin(tweets1_data_list)]


# In[37]:


import numpy as np


# In[38]:


get_ipython().system('pip install -U textblob')


# In[39]:


get_ipython().system('pip install wordcloud')


# In[40]:


from textblob import TextBlob


# In[41]:


from wordcloud import WordCloud


# In[42]:


import re


# In[43]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[55]:


# Data cleaning ie @s, hashtags, links, redundant & irrelevant keywords like covid

def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)   #Remove @s
    text = re.sub(r'#', '', text)    #Removes hashtags
    text = re.sub(r'RT[\s]+', '', text)   #Remove quoted RTs
    text = re.sub(r'https?:\/\/\S+', '', text)   #Remove links
    text = re.sub(r'vaccine', '', text)
    text = re.sub(r'man', '', text)
    text = re.sub(r'COVID','', text)
    text = re.sub(r'Covaxin','', text)
    text = re.sub(r'COVID19','', text)
    text = re.sub(r'vaccinated','', text)
    text = re.sub(r'day','', text)
    text = re.sub(r'year','', text)
    
    
    return text

tweets['text'] = tweets['text'].apply(cleanTxt)


# In[56]:


#Calculate subjectivity & polarity by creating functions for each, using TextBlob library

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

tweets['subjectivity'] = tweets['text'].apply(getSubjectivity)
tweets['polarity'] = tweets['text'].apply(getPolarity)


# In[57]:


#Word cloud visualization

allwords = ' '.join( [text for text in tweets['text']] )
wordcloud = WordCloud(width= 500, height=300, random_state=21, max_font_size=110).generate(allwords)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[60]:


#Compute polarity analysis

def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
    
tweets['analysis'] = tweets['polarity'].apply(getAnalysis)


# In[58]:


#Plotting polarity & subjectivity

x= tweets['polarity']
y= tweets['subjectivity']

plt.figure(figsize=(8,6))
plt.scatter(x, y, c=x, cmap='twilight_shifted', edgecolor='black', linewidth=1, alpha=0.75)
cbar= plt.colorbar()
cbar.set_label('Polarity')
plt.title("What's the spread of sentiments from sampled tweets?")
plt.show()


# In[64]:


# Percentage of poz & tweets

ptweets = tweets[tweets.analysis == 'Positive']
ptweets = ptweets['text']

print("% positive tweets:", round(ptweets.shape[0] / tweets.shape[0]*100, 2))
#74% of polarized tweets are positive


# In[65]:


ntweets = tweets[tweets.analysis == 'Negative']
ntweets = ntweets['text']

print("% negative tweets:", round(ntweets.shape[0] / tweets.shape[0]*100, 2))

#26% of polarized tweets are negative


# In[66]:


neutraltweets = tweets[tweets.analysis == 'Neutral']
neutraltweets = neutraltweets['text']

print("% neutral tweets:", round(neutraltweets.shape[0] / tweets.shape[0]*100, 2))
#large percentage of sampled tweets are neutral


# In[67]:


tweets['analysis'].value_counts()
plt.title("What's the sentiment ponderation across the sampled tweets?")
plt.xlabel('Sentiment')
plt.ylabel('Ponderation')
tweets['analysis'].value_counts().plot(kind='bar')
plt.show()


# In[30]:


### CNN headlines sentiment analysis
import pandas
cnndata = pandas.read_excel("news_headlines.xlsx")
import pandas as pd
cnndata = pd.read_excel("news_headlines.xlsx")


# In[31]:


# Divide headlines database by month from dec20 to oct21

dec20 = cnndata.loc[(cnndata['DATE'] <= 'Dec 31, 2020') & (cnndata['DATE'] >= 'Dec 1, 2020'), :]
jan21 = cnndata.loc[(cnndata['DATE'] <= 'Jan 31, 2021') & (cnndata['DATE'] >= 'Jan 1, 2021'), :]
feb21 = cnndata.loc[(cnndata['DATE'] <= 'Feb 31, 2021') & (cnndata['DATE'] >= 'Feb 1, 2021'), :]
mar21 = cnndata.loc[(cnndata['DATE'] <= 'Mar 31, 2021') & (cnndata['DATE'] >= 'Mar 1, 2021'), :]
apr21 = cnndata.loc[(cnndata['DATE'] <= 'Apr 31, 2021') & (cnndata['DATE'] >= 'Apr 1, 2021'), :]
may21 = cnndata.loc[(cnndata['DATE'] <= 'May 31, 2021') & (cnndata['DATE'] >= 'May 1, 2021'), :]
jun21 = cnndata.loc[(cnndata['DATE'] <= 'Jun 31, 2021') & (cnndata['DATE'] >= 'Jun 1, 2021'), :]
jul21 = cnndata.loc[(cnndata['DATE'] <= 'Jul 31, 2021') & (cnndata['DATE'] >= 'Jul 1, 2021'), :]
aug21 = cnndata.loc[(cnndata['DATE'] <= 'Aug 31, 2021') & (cnndata['DATE'] >= 'Aug 1, 2021'), :]
sep21 = cnndata.loc[(cnndata['DATE'] <= 'Sep 31, 2021') & (cnndata['DATE'] >= 'Sep 1, 2021'), :]
oct21 = cnndata.loc[(cnndata['DATE'] <= 'Oct 31, 2021') & (cnndata['DATE'] >= 'Oct 1, 2021'), :]


# In[32]:


#Calculate HEADLINES subjectivity & polarity 

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

cnndata['h_subjectivity'] = cnndata['HEADLINES'].apply(getSubjectivity)
cnndata['h_polarity'] = cnndata['HEADLINES'].apply(getPolarity)


# In[189]:


#Calculate DESCRIPTIONS subjectivity & polarity 

def cleanTxt(text):
    text = re.sub(r'vaccine', '', text)
    text = re.sub(r'Covid','', text)
    text = re.sub(r'pandemic','', text)  
    text = re.sub(r'Biden','', text)  

    return text


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

cnndata['d_subjectivity'] = cnndata['DESCRIPTIONS'].apply(getSubjectivity)
cnndata['d_polarity'] = cnndata['DESCRIPTIONS'].apply(getPolarity)


# In[72]:


#Headlines Word cloud viz

h_words = ' '.join( [text for text in cnndata['HEADLINES']] )
h_cloud = WordCloud(width= 500, height=300, random_state=21, max_font_size=110).generate(h_words)

plt.imshow(h_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[73]:


#Descriptions Word cloud viz

d_words = ' '.join( [text for text in cnndata['DESCRIPTIONS']] )
d_cloud = WordCloud(width= 500, height=300, random_state=21, max_font_size=110).generate(d_words)

plt.imshow(d_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[74]:


dec20.mean()
jan21.mean()
mar21.mean()
apr21.mean()
may21.mean()
jun21.mean()
jul21.mean()
aug21.mean()
sep21.mean() 
oct21.mean()


# In[75]:


import pandas as pd
newsanalysis = pd.read_excel('newsanalysis.xlsx')
newsanalysis.rename({'Unnamed: 0':'month'}, axis=1, inplace=True)


# In[76]:


x= newsanalysis.month
y= newsanalysis.d_polarity

plt.figure(figsize=(10,6))
plt.plot(x,y, linewidth= 2)
plt.title("How has the polarity of CNN news descriptions evolved from December 2020 to October 2021?")
plt.show()


# In[77]:


x= newsanalysis.month
y= newsanalysis.h_polarity

plt.figure(figsize=(10,6))
plt.plot(x,y, linewidth= 2)
plt.title("How has the polarity of CNN headlines evolved from December 2020 to October 2021?")
plt.show()

#Range from 0.025 to 0.065 --> despite fluctuations, the range is relatively neutral which is expected from a news outline


# In[78]:


### Tweets monthly polarity

#Breakdown of tweets by month

d20 = tweets.loc[(tweets['date'] <= '2020-12-31') & (tweets['date'] >= '2020-12-01'), :]
j21 = tweets.loc[(tweets['date'] <= '2021-01-31') & (tweets['date'] >= '2021-01-01'), :]
f21 = tweets.loc[(tweets['date'] <= '2021-02-31') & (tweets['date'] >= '2021-02-01'), :]
m21 = tweets.loc[(tweets['date'] <= '2021-03-31') & (tweets['date'] >= '2021-03-01'), :]
a21 = tweets.loc[(tweets['date'] <= '2021-04-31') & (tweets['date'] >= '2021-04-01'), :]
ma21 = tweets.loc[(tweets['date'] <= '2021-05-31') & (tweets['date'] >= '2021-05-01'), :]
ju21 = tweets.loc[(tweets['date'] <= '2021-06-31') & (tweets['date'] >= '2021-06-01'), :]
jl21 = tweets.loc[(tweets['date'] <= '2021-07-31') & (tweets['date'] >= '2021-07-01'), :]
au21 = tweets.loc[(tweets['date'] <= '2021-08-31') & (tweets['date'] >= '2021-08-01'), :]
s21 = tweets.loc[(tweets['date'] <= '2021-09-31') & (tweets['date'] >= '2021-09-01'), :]
o21 = tweets.loc[(tweets['date'] <= '2021-10-31') & (tweets['date'] >= '2021-10-01'), :]

tweetsanalysis = pd.read_excel('tweetsanalysis.xlsx')
tweetsanalysis.drop(['Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)


# In[79]:


x= tweetsanalysis.month
y= tweetsanalysis.polarity

plt.figure(figsize=(10,6))
plt.plot(x,y, linewidth= 2)
plt.title("How has the polarity of the sampled tweets evolved from December 2020 to October 2021?")
plt.show()


# In[80]:


x= tweetsanalysis.month
y1= tweetsanalysis.polarity
y2= newsanalysis.h_polarity

plt.figure(figsize=(10,6))
plt.plot(x,y1, label="Tweets polarity", linewidth= 2)
plt.plot(x,y2, label= "CNN headlines polarity", linewidth= 2)

plt.legend()
plt.show()


# In[81]:


x= tweetsanalysis.month
y1= tweetsanalysis.polarity
y2= newsanalysis.h_polarity
y3= newsanalysis.d_polarity

plt.figure(figsize=(10,6))
plt.plot(x,y1, label="Tweets polarity", linewidth= 2)
plt.plot(x,y2, label= "News headlines polarity", linewidth= 2)
plt.plot(x,y3, label= "News descriptions", linewidth= 2)

plt.legend()
plt.show()


# In[83]:


df = pd.read_excel('df.xlsx')
df.drop(['news_polarity'], axis=1, inplace=True)


# In[84]:


#since we're working with timeseries, the percentage change from one period to the next was calculated to compute correlations
df['tweets_change'] = df['t_polarity'].pct_change()
df['headlines_change'] = df['h_polarity'].pct_change()

corr= round(df['headlines_change'].corr(df['tweets_change'], method = 'pearson'), 1)
print("Correlation =", corr)

#correlation is positive but weak --> the percentage change of sample tweets' polarity from one month to the next is slightly correlated with the percentage change of that of CNN's news headlines


# In[85]:


#correlation between the polarity of tweets & news descriptions
df['tweets_change'] = df['t_polarity'].pct_change()
df['descriptions_change'] = df['d_polarity'].pct_change()

corr= round(df['descriptions_change'].corr(df['tweets_change'], method = 'pearson'), 1)
print("Correlation =", corr)


# In[87]:


import seaborn as sns

#sns.set(style="ticks", color_codes=True)    
#g = sns.pairplot(df)
#plt.show()


# In[103]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

x= df.month
y1= df['tweets_change']
y2= df['descriptions_change']

fig, ax1= plt.subplots()
ax2= ax1.twinx()

curve1= ax1.plot(x, y1, label="Public opinion polarity change", color="yellow")
curve2= ax2.plot(x, y2, label="News polarity change", color="red")

plt.plot()
plt.show()


# In[106]:


### Predicting whether Pfizer stock price will increase or deacrease based on CNN headlines

get_ipython().system('pip install vadersentiment')


# In[6]:


import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install yfinance --upgrade --no-cache-dir')


# In[2]:


#Importing stock price data from Yahoo Finance
import yfinance as yf
import datetime as dt

tick1= "PFE"
tick2= "MRNA"
tick3= "JNJ"

stock_PFE= yf.download(tick1, start = "2020-12-1", end="2021-10-31")
stock_MRNA= yf.download(tick2, start = "2020-12-1", end="2021-10-31")
stock_JNJ= yf.download(tick3, start= "2020-12-1", end="2021-10-31")


# In[3]:


stock_PFE
stock_JNJ


# In[4]:


#Create column with month indexes
stock_PFE['Date'] = stock_PFE.index
stock_MRNA['Date'] = stock_MRNA.index
stock_JNJ['Date'] = stock_JNJ.index

stock_PFE['Month'] = stock_PFE['Date'].dt.month
stock_MRNA['Month'] = stock_MRNA['Date'].dt.month
stock_JNJ['Month'] = stock_JNJ['Date'].dt.month

stock_MRNA


# In[14]:


#Calculate average stock price for every month
PFE = stock_PFE.groupby(by="Month")['Adj Close'].mean()
MRNA = stock_MRNA.groupby(by="Month")['Adj Close'].mean()
JNJ = stock_JNJ.groupby(by="Month")['Adj Close'].mean()

final_df = pd.read_excel('final_df.xlsx')
final_df.drop(['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10'], axis=1, inplace=True)
final_df


# In[19]:


#Correlation between the polarity of public opinion and Moderna's stock price 
final_df['tweets_change'] = final_df['t_polarity'].pct_change()
final_df['MRNA_change'] = final_df['MRNA'].pct_change()

corr= round(final_df['tweets_change'].corr(final_df['MRNA_change'], method = 'pearson'), 1)
print("Correlation =", corr)


# In[20]:


#Correlation between the polarity of public opinion and Pfizer's stock price 
final_df['tweets_change'] = final_df['t_polarity'].pct_change()
final_df['PFE_change'] = final_df['PFE'].pct_change()

corr= round(final_df['tweets_change'].corr(final_df['PFE_change'], method = 'pearson'), 1)
print("Correlation =", corr)


# In[86]:


#Correlation between the polarity of public opinion and Johnson & Johnson's stock price 
final_df['tweets_change'] = final_df['t_polarity'].pct_change()
final_df['JNJ_change'] = final_df['JNJ'].pct_change()

corr= round(final_df['tweets_change'].corr(final_df['JNJ_change'], method = 'pearson'), 1)
print("Correlation =", corr)


# In[119]:


##Scatter plots news polarity x stock price
import matplotlib.pyplot as plt

#Pfizer
z= final_df['PFE_change']
w= final_df['JNJ_change']
y= final_df['tweets_change']
x = final_df['news_change']
# d= final_df['month']
   
#pfizer    
plt.figure(figsize=(10,8))
plt.scatter(x, y, c=z, cmap=plt.cm.get_cmap("Greys",5), edgecolor='black')
plt.scatter(x, y, c=w, cmap=plt.cm.get_cmap("Greys",5), edgecolor='black')
plt.xlabel('News polarity')
plt.ylabel('Public opinion')
cbar= plt.colorbar()
cbar.set_label('Stock change')
plt.title('Pfizer')

#jnj
plt.figure(figsize=(10,8))
plt.scatter(x, y, c=w, cmap=plt.cm.get_cmap("Greys",5), edgecolor='black')
plt.xlabel('News polarity')
plt.ylabel('Public opinion')
cbar= plt.colorbar()
cbar.set_label('Stock change')
plt.title('JNJ')


# plt.scatter(z,y2, alpha=1)

# plt.scatter(z, y1, s=x, cmap='twilight_shifted', edgecolor='black', linewidth=1, alpha=1)
# plt.title("What's the spread of sentiments from sampled tweets?")


plt.show()


# In[87]:


final_df['news_change'] = final_df['d_polarity'].pct_change()
final_df

