#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RussianStemmer, EnglishStemmer
from gensim import corpora
import gensim
from operator import itemgetter
import codecs
import csv

# Loading messages scapped from a Facebook group
csvFile = codecs.open('Facebook_group_messages.csv','rU','cp1251')
df = pd.read_csv(csvFile, sep=';')
df = df.fillna('')
dirty = df['Повідомлення'].tolist()
clean = []
# Loading English, Russian and Ukrainian stopwords
csvFile = codecs.open('Stopwords.csv','rU','cp1251')
df = pd.read_csv(csvFile, sep=';',header=None)
stop = df[0].tolist()
stop = "\\b|\\b".join(stop)
# Cleaning texts:
for t in dirty:
    t = str(t)
    t=t.lower()
    t = re.sub(" ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)", ' ', t)
    t = re.sub("@\w+ ?", ' ', t)
    t = re.sub("[^\w\s]|[\d]", ' ', t)
    t = re.sub(stop, ' ', t)
    t = re.sub("\s+", ' ', t)
    clean.append(t)
# Tokenizing and stemming words in texts
flda = []
tokenizer = RegexpTokenizer(r'\w+')
for t in clean:
    tokens = tokenizer.tokenize(t)
    flda.append(tokens)
stemmed1 = []
r_stemmer = RussianStemmer()
e_stemmer = EnglishStemmer()
for t in clean:
    tokens = tokenizer.tokenize(t)
    stemmed_tokens = [e_stemmer.stem(i) for i in tokens]
    stemmed_tokens1 = [r_stemmer.stem(i) for i in stemmed_tokens]
    stemmed1.append(stemmed_tokens1)
stemmed = []
for t in stemmed1:
    tex = " ".join(t)
    stemmed.append(tex)

# Creating and applying 5 topics LDA model:
dictionary = corpora.Dictionary(flda)
corpus = [dictionary.doc2bow(text) for text in flda]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5,
                                           id2word=dictionary, passes=50)
# Display 10 most important keywords for each topic,
# which may be used to name these topics:
print(ldamodel.print_topics(num_topics=5, num_words=10))
topics = []
for i in range(len(flda)):
    l = ldamodel[corpus[i]]
    maxTopic = max(l, key=itemgetter(1))[0]
    topics.append(maxTopic)
# For each text save a topic which dominates in it. In two other
# columns we save clean texts (see row 25) and stemmed texts (see row
# 35).
headers=["Topic index", 'Clean text', 'Stemmed text']
rows = zip(topics,clean,stemmed)
with open('S:/path/LDA topics.csv', 'w', newline='',
          encoding='cp1251') as csvfile:
    wr = csv.writer(csvfile, delimiter=';',
                            quotechar='"',)
    wr.writerow(headers)
with open('S:/path/LDA topics.csv', 'a', newline='',
          encoding='cp1251') as csvfile:
    wr = csv.writer(csvfile, delimiter=';',
                            quotechar='"',)
    for row in rows:
        wr.writerow(row)