#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk import word_tokenize
from textstat import textstat
import numpy as np


# In[123]:


nltk.download('stopwords')
nltk.download('punkt')


# In[124]:


data = {
    'Question': [
        'What is the capital of France?',
        'What is the capital of US?',
        'Explain the theory of relativity in simple terms.',
        'Solve the equation: 2x + 3 = 7.',
        'Describe the process of photosynthesis.',
        'Who discovered penicillin?',
        'Discuss the implications of quantum computing on cryptography.',
        'Write a Python script to sort a list.',
        'Define the term "Artificial Intelligence".',
        'Explain Newton’s Laws of Motion.',
        'Translate the sentence "Hello, how are you?" to French.'
    ],
    'No of Attendees for that Question': [100, 91, 90, 80, 120, 95, 85, 70, 110, 60, 75],
    'No of Students Succeeded': [10, 20, 30, 50, 15, 45, 40, 20, 5, 55, 35]
}



# In[125]:


df = pd.DataFrame(data)


# In[126]:


df['ReadabilityScore'] = df['Question'].apply(textstat.flesch_reading_ease)


# In[127]:


df['QuestionWordCount'] = df['Question'].apply(lambda x: len(word_tokenize(x)))


# In[128]:


df['DifficultyScore'] = (df['No of Students Succeeded'] / df['No of Attendees for that Question'])


# In[129]:


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Question'])
similarity_scores = cosine_similarity(tfidf_matrix)


# In[130]:


most_similar = []
for idx, row in enumerate(similarity_scores):
    row[idx] = -np.inf  
    most_similar_idx = np.argmax(row)
    most_similar.append({
        'MostSimilarQuestion': df['Question'].iloc[most_similar_idx],
        'SimilarityScore': row[most_similar_idx]
    })


# In[131]:


df['ReadabilityRank'] = df['ReadabilityScore'].rank(ascending=False)
df['DifficultyRank'] = df['DifficultyScore'].rank(ascending=False)
df['QuestionRankByWords'] = df['QuestionWordCount'].rank(ascending=False)
df['MostSimilarQuestion'] = [item['MostSimilarQuestion'] for item in most_similar]
df['SimilarityScore'] = [item['SimilarityScore'] for item in most_similar]


# In[132]:


def Readability_categorize_question(score):
    if score >= 70:
        return 'Easy'
    elif score >= 25:
        return 'Medium'
    else:
        return 'Hard'

df['ReadabilityCategory'] = df['ReadabilityScore'].apply(Readability_categorize_question)


# In[133]:


def Difficulty_categorize_question(score):
    if score >= 0.7:
        return 'Hard'
    elif score >= 0.2:
        return 'Medium'
    else:
        return 'Easy'

df['DifficultyCategory'] = df['DifficultyScore'].apply(Difficulty_categorize_question)


# In[134]:


def WordCount_categorize_question(score):
    if score >= 8:
        return 'Large Question'
    elif score >= 5:
        return 'Medium Question'
    else:
        return 'Small Question'

df['QuestionSizebyWord'] = df['QuestionWordCount'].apply(WordCount_categorize_question)


# In[135]:


df[['Question', 'ReadabilityScore', 'DifficultyScore', 'ReadabilityRank', 'ReadabilityCategory', 'DifficultyRank', 'DifficultyCategory','QuestionWordCount','DifficultyRank','QuestionRankByWords','QuestionSizebyWord','MostSimilarQuestion','SimilarityScore']]


# In[ ]:




