#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from google.colab import files
uploaded = files.upload()


# In[3]:


import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.utils import shuffle

import re

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[4]:


get_ipython().system('pip install lime')


# In[5]:


import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


# In[6]:


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


# In[7]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# In[8]:


#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.sentiment.util import *
#from textblob import TextBlob
#from nltk import tokenize
df = pd.read_csv('amazon_vfl_reviews.csv')
df.head()


# In[9]:


df.drop_duplicates(subset ="review", keep = "first", inplace = True)


# In[10]:


df.shape


# In[11]:


df['review'] = df['review'].astype('str')


# In[12]:


import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, log_loss
import gensim
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore

#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[13]:


get_ipython().system('pip install vader')


# In[14]:


import vader


# In[15]:


get_ipython().system('pip install vaderSentiment')


# In[16]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[17]:


reviews_df = pd.read_csv('amazon_vfl_reviews.csv', encoding="UTF-8")


# In[18]:


# CORPUS: CREATE DICTIONARY TO COUNT THE WORDS
count_dict_alex = {}
for doc in df['review']:
    for word in doc.split():
        if word in count_dict_alex.keys():
            count_dict_alex[word] +=1
        else:
            count_dict_alex[word] = 1
            
for key, value in sorted(count_dict_alex.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))


# In[19]:


# REMOVE WORDS THAT OCCUR LESS THAN 10 TIMES
low_value = 10
bad_words = [key for key in count_dict_alex.keys() if count_dict_alex[key] < low_value]
# CREATE A LIST OF LISTS - EACH DOCUMENT IS A STRING BROKEN INTO A LIST OF WORDS
corpus = [doc.split() for doc in df['review']]
clean_list = []
for document in corpus:
    clean_list.append([word for word in document if word not in bad_words])
clean_list


# In[20]:


import gensim
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords


# In[21]:


#This can be changed
clean_list[0][:5]


# In[22]:


# CREATE THE INPUTS OF LDA MODEL: DICTIONARY AND CORPUS
corpora_dict = corpora.Dictionary(clean_list)
corpus = [corpora_dict.doc2bow(line) for line in clean_list]


# In[23]:


# TRAIN THE LDA MODEL
lda_model = LdaModel(corpus=corpus,
                         id2word=corpora_dict,
                         random_state=100,
                         num_topics=3,
                         passes=5,
                         per_word_topics=True)

# See the topics
lda_model.print_topics(-1)


# In[24]:


analyser = SentimentIntensityAnalyzer()


# In[25]:


def sentimentScore(sentences):
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print(str(vs))
        results.append(vs) 
    return results


# In[26]:


sentiment = sentimentScore(df['review'])


# In[27]:


sentiment_df = pd.DataFrame(sentiment)
sentiment_df.head()


# In[28]:


# align index to copy rating column for joining
df.index = sentiment_df.index
sentiment_df['rating'] = df['rating']
echo_vader = pd.concat([df, sentiment_df], axis=1)
echo_vader.head()


# In[29]:


#This needs more changes!!!!!!!!!!!!!!!!!!!!
#Can be created for negative and neutral as well. But need to get the graphs fixed first


#postive sentiment
color = ['#63ace5']
ax = echo_vader.groupby("name").pos.mean().plot.bar(color = color, figsize = (9, 6))

plt.title('Positive Sentiment', fontsize = 20, weight='bold')

# plt.xlabel('Variation', fontsize = 16, weight='bold')
plt.xticks(rotation='90', fontsize=14, weight='bold')
ax.xaxis.label.set_visible(False)

plt.ylabel('Sentiment Rating', fontsize=16, weight='bold')
ax.set_ylim([0,0.5])
plt.yticks(fontsize=14)


ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)


fig = plt.gcf()
plt.show()
plt.draw()

fig.savefig('postive_sentiment.jpg')
("")


# In[30]:


#This needs more changes!!!!!!!!!!!!!!!!!!!!
group = df.groupby('rating').count()
group['date']

color = plt.cm.bone(np.linspace(0, 1, 6))
ax = group['date'].plot.bar(color='#7c86ac', figsize = (10, 6))

plt.title('Echo Ratings', fontsize = 20, weight='bold')
plt.xlabel('Ratings', fontsize = 16, weight='bold')
plt.ylabel('Count', fontsize=16, weight='bold')

plt.xticks(rotation='0', fontsize=14)
plt.yticks(fontsize=14)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.show()

#change colour


# In[ ]:




