#!/usr/bin/env python
# coding: utf-8

# # Project 3
# 
# 
# # Movie Genre Classification
# 
# Classify a movie genre based on its plot.
# 
# <img src="moviegenre.png"
#      style="float: left; margin-right: 10px;" />
# 
# 
# 
# 
# https://www.kaggle.com/c/miia4201-202019-p3-moviegenreclassification/overview
# 
# ### Data
# 
# Input:
# - movie plot
# 
# Output:
# Probability of the movie belong to each genre
# 
# 
# ### Evaluation
# 
# - 20% API
# - 30% Report with all the details of the solution, the analysis and the conclusions. The report cannot exceed 10 pages, must be send in PDF format and must be self-contained.
# - 50% Performance in the Kaggle competition (The grade for each group will be proportional to the ranking it occupies in the competition. The group in the first place will obtain 5 points, for each position below, 0.25 points will be subtracted, that is: first place: 5 points, second: 4.75 points, third place: 4.50 points ... eleventh place: 2.50 points, twelfth place: 2.25 points).
# 
# • The project must be carried out in the groups assigned for module 4.
# • Use clear and rigorous procedures.
# • The delivery of the project is on July 12, 2020, 11:59 pm, through Sicua + (Upload: the API and the report in PDF format).
# • No projects will be received after the delivery time or by any other means than the one established. 
# 
# 
# 
# 
# ### Acknowledgements
# 
# We thank Professor Fabio Gonzalez, Ph.D. and his student John Arevalo for providing this dataset.
# 
# See https://arxiv.org/abs/1702.01992

# ## Sample Submission

# In[1]:


import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split


# In[2]:


dataTraining = pd.read_csv('https://github.com/albahnsen/AdvancedMethodsDataAnalysisClass/raw/master/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
dataTesting = pd.read_csv('https://github.com/albahnsen/AdvancedMethodsDataAnalysisClass/raw/master/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)


# In[3]:


dataTraining.head()


# In[4]:


dataTesting.head()


# In[5]:


get_ipython().run_cell_magic('time', '', '\nimport re\ndef pre_process(text):\n    # lowercase\n    text = text.lower()\n    # tags\n    text = re.sub(\'&lt;/?.*?&gt;\',\' &lt;&gt; \',text)\n    # special characters and digits\n    text=re.sub(\'(\\\\d|\\\\W)+\',\' \',text)\n    # remove punctuation\n    #text = re.sub(\'[.;:!\\\'?,\\"()\\[\\]]\', \'\', text)\n    #text = [REPLACE.sub(\'\', line) for line in text]\n    \n    return text\ndataTraining[\'plot_low\']=dataTraining[\'plot\'].apply(lambda x:pre_process(x))\n\nimport nltk\nnltk.corpus.stopwords.words(\'english\')\nnltk.download(\'wordnet\') \nfrom nltk.corpus import stopwords\n\n\nenglish_stop_words=stopwords.words(\'english\')\ndef remove_stop_words(corpus):\n    removed_stop_words = []\n    for review in corpus:\n        removed_stop_words.append(\n            \' \'.join([word for word in review.split() \n                      if word not in english_stop_words])\n        )\n    return removed_stop_words\n\ndataTraining[\'plot_low_rm\'] = remove_stop_words(dataTraining[\'plot_low\'])\n\n\nfrom nltk.stem.porter import PorterStemmer\ndef get_stemmed_text(corpus):\n    stemmer = PorterStemmer()\n    return [\' \'.join([stemmer.stem(word) for word in review.split()]) for review in corpus]\n\ndataTraining[\'plot_low_rm_stem\'] = get_stemmed_text(dataTraining[\'plot_low_rm\'])')


# In[6]:


dataTraining[['plot','plot_low','plot_low_rm','plot_low_rm_stem']]


# ### Create count vectorizer
# 

# In[7]:


vect = CountVectorizer(ngram_range=(1,2),lowercase=True,max_features=10000)

X_dtm = vect.fit_transform(dataTraining['plot_low_rm_stem'])
X_dtm.shape


# In[26]:


X_dtm.shape[1]


# In[8]:


print(vect.get_feature_names()[:50])


# ### Create y

# In[9]:


dataTraining['genres'] = dataTraining['genres'].map(lambda x: eval(x))

le = MultiLabelBinarizer()
y_genres = le.fit_transform(dataTraining['genres'])


# In[24]:


y_genres


# # Red Neuronal

# In[27]:


model = Sequential()
model.add(Embedding(X_dtm.shape[1] + 1, 128)
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# In[ ]:


X_train, X_test, y_train_genres, y_test_genres = train_test_split(X_dtm, y_genres, test_size=0.33, random_state=42)


# In[ ]:


model.fit(X_train,y_train_genres, validation_data=[X_test, y_test_genres], 
          batch_size=128, epochs=10, verbose=1)
          #callbacks=[PlotLossesKeras()])


# In[ ]:


from sklearn.metrics import accuracy_score
pred = model.predict_classes(X_test)
accuracy_score(pred, y_test_genres)

