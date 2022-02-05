#!/usr/bin/env python
# coding: utf-8

# # Detection of Americans Behavior toward Islam on Facebook
# #### Imports

# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string as st
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
import sklearn.model_selection as model_selection
from sklearn.model_selection import RepeatedStratifiedKFold
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# #### Read Data

# In[55]:


page_names_df = pd.read_csv('fb_news_pagenames.csv')
page_names_df.head()


# In[56]:


posts_df = pd.read_csv('fb_news_posts_20K.csv')
posts_df.head()


# In[57]:


comments_df = pd.read_csv('fb_news_comments_1000K_hashed.csv')
comments_df.head()


# ####  Data Wrangling and Cleaning

# In[58]:


'''
Check For Correlation
'''
posts_df.corr()


# In[59]:


'''
Drop Columns
'''
posts_df.drop(['description','link','created_time','scrape_time'] , axis = 1 , inplace = True)
comments_df.drop(['created_time'] , axis = 1 , inplace = True)


# In[60]:


'''
Rename Columns
''' 
comments_df.rename({'post_name':'post_id'} , axis=1 , inplace = True)
posts_df.rename({'message':'post_message'} , axis=1 , inplace = True)
comments_df.rename({'message':'comment_message'} , axis=1 , inplace = True)


# In[61]:


'''
Check NULLs
'''
print( f'Pages:\n{page_names_df.isnull().sum()}'  )
print( f'Posts:\n{posts_df.isnull().sum()}'  )
print( f'Comments:\n{comments_df.isnull().sum()}'  )


# In[62]:


'''
Drop Null Rows If Message iS NULL
'''
posts_df.dropna( subset=['post_message'] , inplace=True)
comments_df.dropna( subset=['comment_message'] , inplace=True)


# In[63]:


'''
Find Posts about Muslims in age of trump in 2017 Only
'''
words1 = ['sala', 'Arab' , 'Jihad' , 'Islam', 'kaaba' , 'sufi' , 'kaaba','mecca']
posts_df['post_message'] = posts_df['post_message'].str.lower()
posts_df = posts_df.loc[( ( posts_df['post_message'].str.contains('sala') ) | ( posts_df['post_message'].str.contains('Arab') ) | ( posts_df['post_message'].str.contains('arab') ) | ( posts_df['post_message'].str.contains('fethullah') ) | ( posts_df['post_message'].str.contains('palestinian') ) | ( posts_df['post_message'].str.contains('ahmed') ) | ( posts_df['post_message'].str.contains('religious') ) | ( posts_df['post_message'].str.contains('Medina') ) | ( posts_df['post_message'].str.contains('sufi') ) | ( posts_df['post_message'].str.contains('salam') ) | ( posts_df['post_message'].str.contains('mohammad') ) | ( posts_df['post_message'].str.contains('kaaba') ) | ( posts_df['post_message'].str.contains('allah') ) | ( posts_df['post_message'].str.contains('makka') ) | ( posts_df['post_message'].str.contains('mecca') ) | ( posts_df['post_message'].str.contains('saddam') ) | ( posts_df['post_message'].str.contains('ramadan') ) | ( posts_df['post_message'].str.contains('muhammad') ) | ( posts_df['post_message'].str.contains('prophet') ) | ( posts_df['post_message'].str.contains('shia') ) | ( posts_df['post_message'].str.contains('halal') ) | ( posts_df['post_message'].str.contains('jerusalem') ) | ( posts_df['post_message'].str.contains('koran') ) | ( posts_df['post_message'].str.contains('quran') ) | ( posts_df['post_message'].str.contains('Imam') ) | ( posts_df['post_message'].str.contains('sunn') ) | ( posts_df['post_message'].str.contains('mosque') ) | ( posts_df['post_message'].str.contains('muslim') ) |  ( posts_df['post_message'].str.contains('islam') ) |  ( posts_df['post_message'].str.contains('jihad') ) | ( posts_df['post_message'].str.contains('pray') ) | ( posts_df['post_message'].str.contains('iraq') ) | ( posts_df['post_message'].str.contains('sharia') ) | ( posts_df['post_message'].str.contains('hamas') ) ) ]


# In[64]:


len(list(posts_df['post_message']))


# In[65]:


'''
Split Values of column post_id , to match both dfs
'''
posts_df['post_id'] = posts_df['post_id'].str.split('_')
comments_df['post_id'] = comments_df['post_id'].str.split('_')


# In[66]:


'''
Take Index 1 of list in column post_id
'''
posts_df['post_id'] = posts_df['post_id'].apply(lambda x: x[1])
comments_df['post_id'] = comments_df['post_id'].apply(lambda x: x[1])


# In[67]:


'''
Merge Pages and Posts Based On page_id
'''
merge_pages_posts = pd.merge(page_names_df , posts_df , on='page_id')
merge_pages_posts.head()


# In[68]:


'''
Group Comments Based On post_id and Set index
'''
merge_posts_comments = pd.merge(merge_pages_posts , comments_df , on = 'post_id').set_index(['page_name' , 'post_message' , 'post_id'])
merge_posts_comments.head()


# In[69]:


'''
Take only USA and UK
# UK -- Muslims news 558 , if trump news in general 2209
# USA -- Muslims 3216 , 130049
# Qatar -- 0 , 299
# nans 
'''
merge_posts_comments = merge_posts_comments.loc[( (merge_posts_comments['country'] == 'UK') | (merge_posts_comments['country'] == 'USA') )]


# In[70]:


'''
Check Number of Sates of USA and UK
'''
groupby_usa = merge_posts_comments.loc[(merge_posts_comments['country'] == 'USA')].groupby('city/state')
groupby_uk = merge_posts_comments.loc[(merge_posts_comments['country'] == 'UK')].groupby('city/state')
print(f'Number of states in USA {len(list(groupby_usa.groups.keys()))}')
print(f'Number of states in UK {len(list(groupby_uk.groups.keys()))}')


# In[71]:


groupby_usa.groups.keys()


# In[72]:


'''
Get Sates Of USA
'''
states = list( groupby_usa.groups.keys() )
states_df = []
len_of_dfs = []
for state in states:
    st_group = groupby_usa.get_group(state)
    states_df.append(st_group)
    len_of_dfs.append(len(st_group))
 # sorted after group by


# In[73]:


states_df[3].index.unique()# 3
states_df[6].index.unique()# 11
states_df[9].index.unique()# 7
states_df[7].index.unique()# 1


# In[74]:


'''
Balancing: Take 773 samples from each of ny , dc , la , texas , the highest states 
'''
la_df = states_df[3].sample(n = 773 , random_state = 1) ## LA
ny_df = states_df[6].sample(n = 773 , random_state = 1) ## NY
dc_df = states_df[9].sample(n = 773 , random_state = 1) ## DC
tex_df =  states_df[7] ## texax


# In[75]:


"""
Remove Symbols from posts and comments
"""

def cleanup_string(str_in):
    '''
    Remove any symbols
    '''
    try:      
        str1 = str_in.replace("’"," ").replace("‘"," ").replace("“"," ").replace("”"," ").replace("–"," ").replace("\n"," ").replace("\r"," ").replace("§"," ")
        #cleanups
        for char in st.punctuation:
            str1 = str1.replace(char, ' ')        

        
    except Exception:
        print('<----',str_in,'---->')
    return str1

la_df = la_df.reset_index(level='post_message')
la_df['comment_message'] = la_df['comment_message'].apply(lambda x: cleanup_string(x))
la_df['post_message'] = la_df['post_message'].apply(lambda x: cleanup_string(x))

ny_df = ny_df.reset_index(level='post_message')
ny_df['comment_message'] = ny_df['comment_message'].apply(lambda x: cleanup_string(x))
ny_df['post_message'] = ny_df['post_message'].apply(lambda x: cleanup_string(x))

dc_df = dc_df.reset_index(level='post_message')
dc_df['comment_message'] = dc_df['comment_message'].apply(lambda x: cleanup_string(x))
dc_df['post_message'] = dc_df['post_message'].apply(lambda x: cleanup_string(x))

tex_df = tex_df.reset_index(level='post_message')
tex_df['comment_message'] = tex_df['comment_message'].apply(lambda x: cleanup_string(x))
tex_df['post_message'] = tex_df['post_message'].apply(lambda x: cleanup_string(x))


# In[76]:


"""
Remove emojis from posts and comments
"""

def cleanup_emojis(str_in):
    '''
    Remove any emoji
    '''
    try:      
        emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emotions
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002500-\U00002BEF"  # chinese char
                u"\U00002702-\U000027B0"
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u"\U00010000-\U0010ffff"
                u"\u2640-\u2642" 
                u"\u2600-\u2B55"
                u"\u200d"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\ufe0f"  # dingbats
                u"\u3030"
                "]+", flags=re.UNICODE)
        str1 = emoji_pattern.sub(r' ', str_in) # no emoji    
    except Exception:
        print('<----',str_in,'---->')
    return str1

la_df['comment_message'] = la_df['comment_message'].apply(lambda x: cleanup_emojis(x))
la_df['post_message'] = la_df['post_message'].apply(lambda x: cleanup_emojis(x))

ny_df['comment_message'] = ny_df['comment_message'].apply(lambda x: cleanup_emojis(x))
ny_df['post_message'] = ny_df['post_message'].apply(lambda x: cleanup_emojis(x))

dc_df['comment_message'] = dc_df['comment_message'].apply(lambda x: cleanup_emojis(x))
dc_df['post_message'] = dc_df['post_message'].apply(lambda x: cleanup_emojis(x))

tex_df['comment_message'] = tex_df['comment_message'].apply(lambda x: cleanup_emojis(x))
tex_df['post_message'] = tex_df['post_message'].apply(lambda x: cleanup_emojis(x))


# In[77]:


"""
Remove Stop words from posts and comments, commented coz it takes time, use the dataframes below instead
"""

def cleanup_stop_words(str_in):
    '''
    Remove any symbols
    '''
    filtered_sentence = ''
    try:      
        text_tokens = word_tokenize(str_in)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        filtered_sentence = (" ").join(tokens_without_sw)
        
    except Exception:
        print('<----',str_in,'---->')
    return filtered_sentence
la_df['comment_message'] = la_df['comment_message'].apply(lambda x: cleanup_stop_words(x))
la_df['post_message'] = la_df['post_message'].apply(lambda x: cleanup_stop_words(x))

ny_df['comment_message'] = ny_df['comment_message'].apply(lambda x: cleanup_stop_words(x))
ny_df['post_message'] = ny_df['post_message'].apply(lambda x: cleanup_stop_words(x))

dc_df['comment_message'] = dc_df['comment_message'].apply(lambda x: cleanup_stop_words(x))
dc_df['post_message'] = dc_df['post_message'].apply(lambda x: cleanup_stop_words(x))

tex_df['comment_message'] = tex_df['comment_message'].apply(lambda x: cleanup_stop_words(x))
tex_df['post_message'] = tex_df['post_message'].apply(lambda x: cleanup_stop_words(x))


# In[78]:


# dc_df['post_message'] # group by posts names again if u want


# In[79]:


"""
Save data frames to csv, to save time
"""
ny_df.to_csv('ny_df1.csv')
la_df.to_csv('la_df1.csv')
dc_df.to_csv('dc_df1.csv')
tex_df.to_csv('tex_df1.csv')


# #### USA Analysis

# In[80]:


'''
Some analysis "before sampling" , draw bar plot for number of comments in each state , result: the highest number of comments in NY
'''
usa_df = merge_posts_comments.loc[(merge_posts_comments['country'] == 'USA')].sort_values(by = 'city/state')
usa_df.dropna( subset = ['city/state'] , inplace = True)
plt.figure(figsize = (15,8))
ax = sns.countplot( x = 'city/state', order = usa_df['city/state'].value_counts().index , data = usa_df, palette = ["#fe9055" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
i = 0
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , usa_df['city/state'].value_counts()[i] , ha = "center" )
    i += 1


# #### Sentiment Analysis

# In[81]:


"""
TextBlob for classify posts  as positive , neutral or negative polarity
"""

def sentiment_analysis(df):
    
    def getSubjectivity(text):
           return TextBlob(text).sentiment.subjectivity
    #Create a function to get the polarity
    def getPolarity(text):
           return TextBlob(text).sentiment.polarity

    #Create two new columns ‘Subjectivity’ & ‘Polarity’
    df['TextBlob_Subjectivity'] = df['post_message'].apply(getSubjectivity)
    df['TextBlob_Polarity'] = df['post_message'].apply(getPolarity)
    
    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
    df ['TextBlob_Analysis'] = df['TextBlob_Polarity'].apply(getAnalysis)
    return df

sentiment_analysis(la_df)
sentiment_analysis(dc_df)
sentiment_analysis(ny_df)
sentiment_analysis(tex_df)


# In[82]:


la_df['TextBlob_Analysis'].unique()


# In[83]:


"""
TextBlob for classify comments  as positive , neutral or negative polarity
"""

def sentiment_analysis(df):
    
    def getSubjectivity(text):
           return TextBlob(text).sentiment.subjectivity
    #Create a function to get the polarity
    def getPolarity(text):
           return TextBlob(text).sentiment.polarity

    #Create two new columns ‘Subjectivity’ & ‘Polarity’
    df['comment_Subjectivity'] = df['comment_message'].apply(getSubjectivity)
    df['comment_Polarity'] = df['comment_message'].apply(getPolarity)
    
    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
    df ['comment_Analysis'] = df['comment_Polarity'].apply(getAnalysis)
    return df

sentiment_analysis(la_df)
sentiment_analysis(dc_df)
sentiment_analysis(ny_df)
sentiment_analysis(tex_df)


# In[84]:


'''
Change values of pos to 1, neg to 0, neu to 2, Mapping categorical labels to numeric
'''

la_df['TextBlob_Analysis'] = la_df['TextBlob_Analysis'].map({'Positive':1, 'Negative':0, 'Neutral':2})
la_df['comment_Analysis'] = la_df['comment_Analysis'].map({'Positive':1, 'Negative':0, 'Neutral':2})
dc_df['TextBlob_Analysis'] = dc_df['TextBlob_Analysis'].map({'Positive':1, 'Negative':0, 'Neutral':2})
dc_df['comment_Analysis'] = dc_df['comment_Analysis'].map({'Positive':1, 'Negative':0, 'Neutral':2})
ny_df['TextBlob_Analysis'] = ny_df['TextBlob_Analysis'].map({'Positive':1, 'Negative':0, 'Neutral':2})
ny_df['comment_Analysis'] = ny_df['comment_Analysis'].map({'Positive':1, 'Negative':0, 'Neutral':2})
tex_df['TextBlob_Analysis'] = tex_df['TextBlob_Analysis'].map({'Positive':1, 'Negative':0, 'Neutral':2})
tex_df['comment_Analysis'] = tex_df['comment_Analysis'].map({'Positive':1, 'Negative':0, 'Neutral':2})


# https://machinelearninggeek.com/sentiment-analysis-using-python/

# In[85]:


'''
count of sentiments of comments in LA
'''

la_df = la_df.sort_values(by = 'comment_Analysis')
plt.figure(figsize = (8,8))
ax = sns.countplot( x = 'comment_Analysis', order = la_df['comment_Analysis'].value_counts().index , data = la_df, palette = ["#fe9055" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
plt.xticks( [0,1,2] , ['Neutral','Positive','Negative']  )
i = 2
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , la_df['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1
    

#plt.show()


la_df = la_df.sort_values(by = 'TextBlob_Analysis')
la_df1 = la_df.drop_duplicates(subset = ['post_message'])
plt.figure(figsize = (8,8))
ax = sns.countplot( x = 'TextBlob_Analysis', order = la_df1['TextBlob_Analysis'].value_counts().index , data = la_df1, palette = ["#fe9055" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
plt.xticks( [0,1,2] , ['Neutral','Positive','Negative']  )
i = 2
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , la_df1['TextBlob_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1
    

plt.show()


# In[86]:


'''
concat all states to predict, they were separated to analysis purpose
'''

list_dfs_comm = [ la_df , ny_df , dc_df , tex_df ]
df_all = pd.concat( list_dfs_comm , sort = False )


# In[87]:


df_all.describe(include = 'all')


# In[88]:


#df_all.index.unique()


# In[89]:


'''
correlation between columns, we need corr with comments analysis
'''

Corr_matrix = df_all.corr()
f,ax = plt.subplots(figsize = (14, 14))
sns.heatmap(Corr_matrix, annot = True, linewidths = .5, fmt = '.1f',ax = ax, cmap = sns.cubehelix_palette(as_cmap = True))


# In[90]:


'''
Drop columns from all data frame
'''
df_all = df_all.drop(['country','from_name','from_id', 'page_id', 'react_angry' , 'react_haha' ,'react_like' , 'react_love' , 'react_sad' , 'react_wow' , 'shares' , 'TextBlob_Polarity' , 'TextBlob_Subjectivity' , 'TextBlob_Analysis'], axis = 1)


# #### Some analysis to find relation between posts and comments regarding to polarity

# In[91]:


'''
take the origins from origin data
LA
1-> today  iraq declared victory over islamic state in mosul after nearly nine months of fighting      here s what we saw when we sent two photographers to mosul 
0-> the rising democrat party star linda sarsour doubles down on her radical call for  jihad  against president trump   
DC
1-> iraq celebrated a major triumph over the islamic state  but at the cost of a city s destruction 
0-> the scale and gravity of the loss of civilian lives during the military operation to retake mosul must immediately be publicly acknowledged at the highest levels of government in iraq and states that are part of the us led coalition   lynn maalouf  director of research for the middle east at amnesty international  said in a statement 
NY
0-> italy s plan to reduce the risk of a jihadi inspired attack is pinned in small part on an imam who bikes to the prison every week and exhorts muslim inmates not to stray from life s  right path  or hate people who aren t muslim 
1-> nobel peace prize winner malala yousafzai meets displaced children at a refugee camp in iraq to highlight their needs and their right to access education  http   abcn ws 2t4ejps
TEX
1-> over the top  linda sarsour calls for jihad on trump at hamas meeting    at a hamas meeting  muslim activist linda sarsour openly calls for jihad against president trump  his entire administration  and the united states government in an act of politicized sedition 
0-> president trump delivered a speech in warsaw that likely confirmed the worst fears of globalists and islamists
'''
list(la_df1.loc[(la_df1['TextBlob_Analysis'] == 1)]['post_message'])


# In[92]:


'''
LA pos post with num of comments pos and negatives
LA neg post with num of comments pos and negatives
'''

dd_la = la_df.loc[((la_df['TextBlob_Analysis'] == 1) & (la_df['post_message'] == 'today  iraq declared victory over islamic state in mosul after nearly nine months of fighting      here s what we saw when we sent two photographers to mosul '))]
plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'comment_Analysis', order = dd_la['comment_Analysis'].value_counts().index[0:2] , data = dd_la, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
ax.set(ylim=(0, 40))
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 0
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , dd_la['comment_Analysis'].value_counts()[i] , ha = "center" )
    i += 1

print(dd_la['comment_Analysis'].value_counts())   
dd_la = la_df.loc[((la_df['TextBlob_Analysis'] == 0) & (la_df['post_message'] == 'the rising democrat party star linda sarsour doubles down on her radical call for  jihad  against president trump   '))]
plt.figure(figsize = (4,4))
# add title from paint withoriginal post text
# negative post
ax = sns.countplot( x = 'comment_Analysis', order = dd_la['comment_Analysis'].value_counts().index[2:0:-1] , data = dd_la, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
ax.set(ylim=(0, 40))
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 1
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , dd_la['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1
    
print(dd_la['comment_Analysis'].value_counts())   
plt.show()


# In[93]:


'''
DC pos post with num of comments pos and negatives
DC neg post with num of comments pos and negatives
'''

dd_la = dc_df.loc[((dc_df['TextBlob_Analysis'] == 1) & (dc_df['post_message'] == 'iraq celebrated a major triumph over the islamic state  but at the cost of a city s destruction '))]
plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'comment_Analysis', order = dd_la['comment_Analysis'].value_counts().index[1:] , data = dd_la, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
ax.set(ylim=(0, 40))
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 1
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , dd_la['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1
    
print(dd_la['comment_Analysis'].value_counts())    
    
dd_la = dc_df.loc[((dc_df['TextBlob_Analysis'] == 0) & (dc_df['post_message'] == ' the scale and gravity of the loss of civilian lives during the military operation to retake mosul must immediately be publicly acknowledged at the highest levels of government in iraq and states that are part of the us led coalition   lynn maalouf  director of research for the middle east at amnesty international  said in a statement '))]
plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'comment_Analysis', order = dd_la['comment_Analysis'].value_counts().index[1:] , data = dd_la, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
ax.set(ylim=(0, 40))
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 1
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , dd_la['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1    
    
print(dd_la['comment_Analysis'].value_counts())


# In[94]:


'''
NY pos post with num of comments pos and negatives
NY neg post with num of comments pos and negatives
'''

dd_la = ny_df.loc[((ny_df['TextBlob_Analysis'] == 0) & (ny_df['post_message'] == 'italy s plan to reduce the risk of a jihadi inspired attack is pinned in small part on an imam who bikes to the prison every week and exhorts muslim inmates not to stray from life s  right path  or hate people who aren t muslim '))]
plt.figure(figsize = (3,4))
ax = sns.countplot( x = 'comment_Analysis', order = dd_la['comment_Analysis'].value_counts().index[1:] , data = dd_la, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
ax.set(ylim=(0, 40))
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 1
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , dd_la['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1
    
print(dd_la['comment_Analysis'].value_counts())    
    
dd_la = ny_df.loc[((ny_df['TextBlob_Analysis'] == 1) & (ny_df['post_message'] == 'nobel peace prize winner malala yousafzai meets displaced children at a refugee camp in iraq to highlight their needs and their right to access education  http   abcn ws 2t4ejps'))]
plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'comment_Analysis', order = dd_la['comment_Analysis'].value_counts().index[1:] , data = dd_la, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
ax.set(ylim=(0, 40))
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 1
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , dd_la['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1    
    
print(dd_la['comment_Analysis'].value_counts())


# In[95]:


'''
Tex pos post with num of comments pos and negatives
Tex neg post with num of comments pos and negatives
'''

dd_la = tex_df.loc[((tex_df['TextBlob_Analysis'] == 0) & (tex_df['post_message'] == 'president trump delivered a speech in warsaw that likely confirmed the worst fears of globalists and islamists'))]
plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'comment_Analysis', order = dd_la['comment_Analysis'].value_counts().index[1:] , data = dd_la, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
ax.set(ylim=(0, 40))
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 1
for p in ax.patches:
    height = p.get_height()
    
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , dd_la['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1
    
print(dd_la['comment_Analysis'].value_counts())    
    
dd_la = tex_df.loc[((tex_df['TextBlob_Analysis'] == 1) & (tex_df['post_message'] == 'over the top  linda sarsour calls for jihad on trump at hamas meeting    at a hamas meeting  muslim activist linda sarsour openly calls for jihad against president trump  his entire administration  and the united states government in an act of politicized sedition '))]
plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'comment_Analysis', order = dd_la['comment_Analysis'].value_counts().index[2:0:-1], data = dd_la, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
ax.set(ylim=(0, 40))
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 1
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , dd_la['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1    
    
print(dd_la['comment_Analysis'].value_counts())


# In[96]:


'''
Equation, avg of positivity to all negative posts in Texas 
'''
def Average(lst):
    return sum(lst) / len(lst)

avg = 0
t = tex_df.copy().loc[(( tex_df['comment_Analysis'] == 0 ) | ( tex_df['comment_Analysis'] == 1 ))]
posts = t.loc[(t.TextBlob_Analysis == 0)].groupby('post_message')
df1 = ''
list_pos = [] # proportion of positives to each post
list( g1.groups.keys() )
for post in g1.groups: 
    dataframe1 = posts.get_group(post)
    lenPos = len(dfataframe1.loc[(dfataframe1.comment_Analysis == 1)])
    lenAll = len(dfataframe1) # total neg and pos
    list_pos.append((lenPos / lenAll) * 100)
    
avg = Average(list_pos)
avg


# In[97]:


'''
Equation, avg of positivity to all pos posts in Texas 
'''
def Average(lst):
    return sum(lst) / len(lst)

avg = 0
t = tex_df.copy().loc[(( tex_df['comment_Analysis'] == 0 ) | ( tex_df['comment_Analysis'] == 1 ))]
g1 = t.loc[(t.TextBlob_Analysis == 1)].groupby('post_message')
df1 = ''
list_pos = [] # proportion of positives to each post
list( g1.groups.keys() )
for c in g1.groups: 
    df1 = g1.get_group(c)
    lenPos = len(df1.loc[(df1.comment_Analysis == 1)])
    lenAll = len(df1) # total neg and pos
    list_pos.append((lenPos / lenAll) * 100)
    
avg = Average(list_pos)
avg


# In[98]:


'''
Equation, avg of positivity to all negative posts in NY
'''
def Average(lst):
    return sum(lst) / len(lst)

avg = 0
t = ny_df.copy().loc[(( ny_df['comment_Analysis'] == 0 ) | ( ny_df['comment_Analysis'] == 1 ))]
g1 = t.loc[(t.TextBlob_Analysis == 0)].groupby('post_message')
df1 = ''
list_pos = [] # proportion of positives to each post
list( g1.groups.keys() )
for c in g1.groups: 
    df1 = g1.get_group(c)
    lenPos = len(df1.loc[(df1.comment_Analysis == 1)])
    lenAll = len(df1) # total neg and pos
    list_pos.append((lenPos / lenAll) * 100)
    
avg = Average(list_pos)
avg


# In[99]:


'''
Equation, avg of positivity to all pos posts in NY
'''
def Average(lst):
    return sum(lst) / len(lst)

avg = 0
t = ny_df.copy().loc[(( ny_df['comment_Analysis'] == 0 ) | ( ny_df['comment_Analysis'] == 1 ))]
g1 = t.loc[(t.TextBlob_Analysis == 1)].groupby('post_message')
df1 = ''
list_pos = [] # proportion of positives to each post
list( g1.groups.keys() )
for c in g1.groups: 
    df1 = g1.get_group(c)
    lenPos = len(df1.loc[(df1.comment_Analysis == 1)])
    lenAll = len(df1) # total neg and pos
    list_pos.append((lenPos / lenAll) * 100)
    
avg = Average(list_pos)
avg


# In[100]:


'''
Equation, avg of positivity to all negative posts in DC
'''
def Average(lst):
    return sum(lst) / len(lst)

avg = 0
t = dc_df.copy().loc[(( dc_df['comment_Analysis'] == 0 ) | ( dc_df['comment_Analysis'] == 1 ))]
g1 = t.loc[(t.TextBlob_Analysis == 0)].groupby('post_message')
df1 = ''
list_pos = [] # proportion of positives to each post
list( g1.groups.keys() )
for c in g1.groups: 
    df1 = g1.get_group(c)
    lenPos = len(df1.loc[(df1.comment_Analysis == 1)])
    lenAll = len(df1) # total neg and pos
    list_pos.append((lenPos / lenAll) * 100)
    
avg = Average(list_pos)
avg


# In[101]:


'''
Equation, avg of positivity to all pos posts in DC
'''
def Average(lst):
    return sum(lst) / len(lst)

avg = 0
t = dc_df.copy().loc[(( dc_df['comment_Analysis'] == 0 ) | ( dc_df['comment_Analysis'] == 1 ))]
g1 = t.loc[(t.TextBlob_Analysis == 1)].groupby('post_message')
df1 = ''
list_pos = [] # proportion of positives to each post
list( g1.groups.keys() )
for c in g1.groups: 
    df1 = g1.get_group(c)
    lenPos = len(df1.loc[(df1.comment_Analysis == 1)])
    lenAll = len(df1) # total neg and pos
    list_pos.append((lenPos / lenAll) * 100)
    
avg = Average(list_pos)
avg


# In[102]:


'''
Equation, avg of positivity to all negative posts in LA
'''
def Average(lst):
    return sum(lst) / len(lst)

avg = 0
t = la_df.copy().loc[(( la_df['comment_Analysis'] == 0 ) | ( la_df['comment_Analysis'] == 1 ))]
g1 = t.loc[(t.TextBlob_Analysis == 0)].groupby('post_message')
df1 = ''
list_pos = [] # proportion of positives to each post
list( g1.groups.keys() )
for c in g1.groups: 
    df1 = g1.get_group(c)
    lenPos = len(df1.loc[(df1.comment_Analysis == 1)])
    lenAll = len(df1) # total neg and pos
    list_pos.append((lenPos / lenAll) * 100)
    
avg = Average(list_pos)
avg


# In[103]:


'''
Equation, avg of positivity to all pos posts in LA
'''
def Average(lst):
    return sum(lst) / len(lst)

avg = 0
t = la_df.copy().loc[(( la_df['comment_Analysis'] == 0 ) | ( la_df['comment_Analysis'] == 1 ))]
g1 = t.loc[(t.TextBlob_Analysis == 1)].groupby('post_message')
df1 = ''
list_pos = [] # proportion of positives to each post
list( g1.groups.keys() )
for c in g1.groups: 
    df1 = g1.get_group(c)
    lenPos = len(df1.loc[(df1.comment_Analysis == 1)])
    lenAll = len(df1) # total neg and pos
    list_pos.append((lenPos / lenAll) * 100)
    
avg = Average(list_pos)
avg


# In[ ]:





# In[104]:


"""
Take only positive and negative to predict (binary classification)
"""

df_all = df_all.loc[(( df_all['comment_Analysis'] == 0 ) | ( df_all['comment_Analysis'] == 1 ))]


# In[105]:


# '''
# Word cloud of posts
# '''
plt.figure(figsize = (10,10))
text = " ".join(cat.split()[1] for cat in df_all.post_message)
wordcloud = WordCloud(collocations = False, background_color = 'white' , colormap = 'RdYlGn').generate(text)# sns.cubehelix_palette(as_cmap = True)
plt.imshow(wordcloud , interpolation = 'bilinear')
plt.axis("off")
plt.show()


# In[106]:


wordcloud.to_file('p2.png')


# In[107]:


# '''
# Word cloud of comments
# '''

plt.figure(figsize = (10,10))
text = " ".join(cat.split()[0] for cat in df_all.comment_message)
wordcloud = WordCloud(collocations = False, background_color = 'white' , colormap = 'RdYlGn').generate(text)
plt.imshow(wordcloud , interpolation = 'bilinear')
plt.axis("off")
plt.show()


# In[108]:


wordcloud.to_file('c1.png')


# In[109]:


'''
check value counts of pos and neg comments
'''

plt.figure(figsize = (4,4))
ax = sns.countplot( x = 'comment_Analysis', order = df_all['comment_Analysis'].value_counts().index , data = df_all, palette = ["#ace1af" , "#fe766a" , "#eb717c" , "#fe5862" , "#a54780" , "#8d2865" ,  "#6b1e47" , "#3d033b" , "#1e0034"] ) # , "" , "#6b1e47"  , "#7b174a" ,
plt.xticks( [0,1] , ['Positive','Negative']  )
i = 1
for p in ax.patches:
    height = p.get_height()
    ax.text( p.get_x() + p.get_width() / 2. , height + 0.1 , df_all['comment_Analysis'].value_counts()[i] , ha = "center" )
    i -= 1


# In[110]:


df_all['comment_Analysis'].value_counts().index


# #### RF , oversampling and splitting data

# In[111]:


'''
stemming post_message
'''
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def stem_fun(str1):
    
    ps = PorterStemmer()
    sentence = str1
    words = word_tokenize(sentence)
    listw = []
    for w in words:
        listw.append(ps.stem(w))
    return listw
        
df_all['post_message'] = df_all['post_message'].apply(lambda x: stem_fun(x))
df_all['post_message']


# In[112]:


'''
stemming comment_message
'''
df_all['comment_message'] = df_all['comment_message'].apply(lambda x: stem_fun(x))
df_all['comment_message']


# In[113]:


'''
lemitization using snowballstemmer posts ( double stemming) as stemming and lemitization
'''
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
def lem_fun(str1):
    
    stemmer = SnowballStemmer("english")
    sentence = str1
    #words = word_tokenize(sentence)
    listw = []
    for w in sentence:
        listw.append(stemmer.stem(w))
    return listw

df_all['post_message'] = df_all['post_message'].apply(lambda x: lem_fun(x))
df_all['post_message']


# In[114]:


df_all['comment_message'] = df_all['comment_message'].apply(lambda x: lem_fun(x))
df_all['comment_message']


# In[115]:


'''
join lists in comments and posts
'''

def tolist(str1):
    return  ' '.join(str1)

df_all['post_message'] = df_all['post_message'].apply(lambda x: tolist(x))
df_all['comment_message'] = df_all['comment_message'].apply(lambda x: tolist(x))


# Remove https://beckernick.github.io/oversampling-modeling/

# In[116]:


'''
Bag of words, count words in comments, To predict comments
'''

token = RegexpTokenizer(r'[a-zA-Z0-9]+')

cv = CountVectorizer( lowercase = True, stop_words = 'english', ngram_range = (1,1), tokenizer = token.tokenize )

text_counts = cv.fit_transform(df_all.comment_message)


# In[117]:


'''
using tf-idf instead of bag of words
'''

tf = TfidfVectorizer()
text_tf = tf.fit_transform( df_all.comment_message )


# In[118]:


'''
Split train and test set
'''

X_train, X_test, y_train, y_test = train_test_split(text_counts, df_all['comment_Analysis'], test_size = 0.3, random_state = 1)


# In[119]:


'''
Split train and test set
'''

X_train1, X_test1, y_train1, y_test1 = train_test_split(text_tf, df_all['comment_Analysis'], test_size = 0.3, random_state = 123)


# In[120]:


"""
Balancing data, over sampling using smote to make class 0 as class 1, and Random Forest using BOW
"""

sm = SMOTE(random_state = 12, sampling_strategy = { 0 :  800 } )
x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
clf = RandomForestClassifier(n_estimators = 25, random_state = 12).fit(x_train_res, y_train_res)
predicted = clf.predict(X_test) 
print("RF Accuracy: ", metrics.accuracy_score(y_test, predicted))#0.7439446366782007  kfold-> .72, if test .2 --> .75
print(classification_report(y_test,predicted))


# In[121]:


"""
Random Forest using TF-IDF
"""

x_train_res, y_train_res = sm.fit_resample(X_train1, y_train1)
clf = RandomForestClassifier(n_estimators = 25, random_state = 12).fit(x_train_res, y_train_res)
predicted = clf.predict(X_test1) 
print("RF Accuracy: ", metrics.accuracy_score(y_test1, predicted))#0.7179930795847751  if test .2 --> .74
print(classification_report(y_test1,predicted))


# #### Train Models (Multinomial  Naive Bayes)

# In[122]:


'''
Build the Text Classification Model using  CountVector(or BoW)
'''

#### Model Generation Using Multinomial Naive Bayes
x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
clf = MultinomialNB().fit( x_train_res, y_train_res )
predicted = clf.predict( X_test )
print("MultinomialNB Accuracy:",metrics.accuracy_score( y_test, predicted ))#0.7352941176470589  kfold .72, if test .2 --> .73
print(classification_report(y_test,predicted))


# In[123]:


'''
Build the Text Classification Model using TF-IDF
'''
x_train_res, y_train_res = sm.fit_resample(X_train1, y_train1)
clf = MultinomialNB().fit(x_train_res, y_train_res)
predicted = clf.predict(X_test1)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test1, predicted))#0.7941176470588235 if test .2 --> .76
print(classification_report(y_test1,predicted))


# #### Train Models (LR)

# In[124]:


"""
using BOW
"""
grid = {"C":np.logspace(-3,3,7), "penalty":["l2", "l1"], "solver":['liblinear','newton-cg']}# l1 lasso l2 ridge

x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
clf =  LogisticRegression()
gs = GridSearchCV(clf, grid, cv = 3, scoring = 'f1_micro').fit(x_train_res, y_train_res)
predicted = gs.predict(X_test)
print("LogisticRegression Accuracy:", metrics.accuracy_score(y_test, predicted))#0.773356401384083   .77 kfold , if test .2 --> .77
print(classification_report(y_test,predicted))


# In[125]:


'''
using tf-idf
'''

x_train_res, y_train_res = sm.fit_resample(X_train1, y_train1)
clf =  LogisticRegression()
gs = GridSearchCV(clf ,grid, cv = 5, scoring = 'f1_micro').fit(x_train_res, y_train_res)
predicted = gs.predict(X_test1)
print("LogisticRegression Accuracy:", metrics.accuracy_score(y_test1, predicted))#0.82  if test .2 --> .82  , final -- >0.84
print(classification_report(y_test1,predicted))


# In[126]:


#define metrics
y_pred_proba = gs.predict_proba(X_test1)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test1,  y_pred_proba)
auc = metrics.roc_auc_score(y_test1, y_pred_proba)

#create ROC curve
plt.plot(fpr,tpr,label = "AUC=" + str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 4)
plt.show()


# #### Train Models (SVM)

# In[127]:


"""
using BOW
"""

parameters = {'C': [1,10,100], 'kernel': ['linear' , 'rbf' , 'poly'],
          'gamma': [0.001, 0.01, 1] }

x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
model = svm.SVC()
grid = GridSearchCV(estimator = model, param_grid = parameters , scoring = 'accuracy')
clf =  grid.fit(x_train_res, y_train_res)
predicted = clf.predict(X_test)
print("SVM Accuracy:", metrics.accuracy_score(y_test, predicted))#0.773356401384083  , if test .2 --> .78
print(classification_report(y_test,predicted))


# In[128]:


'''
using tf-idf
'''
parameters = {'C': [1,10,100], 'kernel': ['linear' , 'rbf' , 'poly'],
          'gamma': [0.001, 0.01, 1] }
x_train_res, y_train_res = sm.fit_resample(X_train1 , y_train1)
model = svm.SVC()
grid = GridSearchCV(estimator = model, param_grid = parameters , scoring = 'accuracy')
clf =  grid.fit(x_train_res , y_train_res)
predicted = clf.predict(X_test1)
print("SVM Accuracy:", metrics.accuracy_score(y_test1, predicted))#0.7906574394463668  if test .2 --> .79
print(classification_report(y_test1,predicted))


# #### Train Models (XGBOOST)

# In[129]:


"""
using BOW
"""

parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 100, 10),
    'learning_rate': [1, 10, 5]
}

x_train_res, y_train_res = sm.fit_resample(X_train, y_train)
estimator =  xgb.XGBClassifier()
clf = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'accuracy',
    verbose=True
).fit(x_train_res, y_train_res)
predicted = clf.predict(X_test)
print("XGB Accuracy:", metrics.accuracy_score(y_test, predicted))#0.801038062283737  if test .2 --> .0.78 
print(classification_report(y_test,predicted))


# In[130]:


'''
using tf-idf
'''

x_train_res, y_train_res = sm.fit_resample(X_train1, y_train1)
estimator =  xgb.XGBClassifier()
clf = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'accuracy',
    verbose=True
).fit(x_train_res, y_train_res)
predicted = clf.predict(X_test1)
print("XGB Accuracy:", metrics.accuracy_score(y_test1, predicted))#0.7439446366782007   if test .2 --> .74
print(classification_report(y_test1,predicted))


# #### KNN 

# In[131]:


'''
using BOW
'''

x_train_res, y_train_res = sm.fit_resample(X_train, y_train)# the sampling is for training only, to train the model in right way
estimator_KNN = KNeighborsClassifier(algorithm = 'auto')
parameters_KNN = {
    'n_neighbors': (1,10, 1),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev', 'euclidean')
}
                   
# with GridSearch
clf = GridSearchCV(
    estimator = estimator_KNN,
    param_grid = parameters_KNN,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
).fit(x_train_res, y_train_res)
predicted = clf.predict(X_test)
print("KNN Accuracy:", metrics.accuracy_score(y_test, predicted))#0.59  if test .2 -->.62
print(classification_report(y_test,predicted))


# In[132]:


"""
using TF-IDF
"""

x_train_res, y_train_res = sm.fit_resample(X_train1, y_train1)# the sampling is for training only, to train the model in right way
estimator_KNN = KNeighborsClassifier(algorithm = 'auto')
parameters_KNN = {
    'n_neighbors': (1,10, 1),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev' , 'euclidean')
}
                   
# with GridSearch
clf = GridSearchCV(
    estimator = estimator_KNN,
    param_grid = parameters_KNN,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
).fit(x_train_res, y_train_res)
predicted = clf.predict(X_test1)
print("KNN Accuracy:", metrics.accuracy_score(y_test1, predicted))#0.671280276816609   if test .2 --> .67
print(classification_report(y_test1,predicted))

