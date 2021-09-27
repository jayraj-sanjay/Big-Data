# Databricks notebook source
pip install nltk

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import RegexpParser
nltk.download('averaged_perceptron_tagger')

# COMMAND ----------

#File location: /FileStore/tables/textFile-1.txt
stop_words = set(stopwords.words('english'))
input = sc.textFile("/FileStore/tables/textFile-1.txt")

# COMMAND ----------

#Split the document and removed stop words
words = input.flatMap(lambda x: x.split()).filter(lambda x: len(x) > 1).filter(lambda x:x not in stop_words)


# COMMAND ----------

#Used NLTK Parts of Speach Tagging to tag each word
tokens_tag = nltk.pos_tag(words.collect())
tagged = sc.parallelize(tokens_tag)

# COMMAND ----------

#Filtered out the words where the tag was NNP - proper noun, singular
nnp_words = tagged.filter(lambda x: x[1]=='NNP')

# COMMAND ----------

#Added one two each word and then summed it up to get the count
wordCounts = nnp_words.map(lambda x:(x[0],1)).reduceByKey(lambda x,y:x+y)

# COMMAND ----------

#Sorted in Descending order on the count
wordCounts.sortBy(lambda x: -x[1]).collect()
