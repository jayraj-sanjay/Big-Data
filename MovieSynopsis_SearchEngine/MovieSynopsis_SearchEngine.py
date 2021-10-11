# Databricks notebook source
pip install nltk

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import math

#reading the input files
searchQueries = sc.textFile("/FileStore/tables/searchQueries.txt").collect()
input1 = sc.textFile("/FileStore/tables/plot_summaries.txt")
input2 = sc.textFile("/FileStore/tables/movie_metadata.tsv")

#pre processing the input rdd's to key value pairs
id_summary = input1.map(lambda x:x.lower().split("\t"))
id_name = input2.map(lambda x:x.split("\t")).map(lambda x:(x[0],x[2]))

stop_words = set(stopwords.words('english'))
N=id_summary.count()

# COMMAND ----------

#Calculating Term Frequency - Splitting the sentence -> removing stop words -> adding one to each pair and then summing it up
map1 = id_summary.flatMap(lambda x: [((x[0],i),1) for i in x[1].split() if i not in stop_words])
reduce1 = map1.reduceByKey(lambda x,y: x+y)
tf=reduce1.map(lambda x: (x[0][1],(x[0][0],x[1])))

# COMMAND ----------

#Calculating IDF for each term in all the documents
idf = reduce1.map(lambda x: (x[0][1], (x[0][0], x[1], 1)))
idf_map = idf.map(lambda x : (x[0], x[1][2]))
idf_doc = idf_map.reduceByKey(lambda x,y : x+y).map(lambda x: (x[0], math.log10(N/x[1])))

# COMMAND ----------

#Joining the TF and IDF rdd and calculating the TF-IDF values
tf_idf = tf.join(idf_doc)
tf_idf_doc = tf_idf.map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1], x[1][0][1]*x[1][1] )))
#Joining the TF-IDF with the movie names 
tf_idf_movies = tf_idf_doc.join(id_name)

# COMMAND ----------

def getMovieForTerm(term):
  return tf_idf_movies.filter(lambda x : x[1][0][0]== term).sortBy(keyfunc = lambda x: x[1][0][3] ,ascending=False).map(lambda x : (x[1][1],x[1][0][3] ))

# COMMAND ----------

#for each query in the search queries text
for query in searchQueries:
  print("Searching for : ", query)
  query_terms = query.lower().split()
  if len(query_terms) > 1:
    #Finding cosine similarity between each word in the query and the documents
    query_rdd= sc.parallelize(query_terms)
    query_words_tf =query_rdd.map(lambda x : (x,1)).reduceByKey(lambda x,y : x+y)
    idf_doc_words= tf_idf_movies.map(lambda x : (x[1][0][0], (x[1][1], x[1][0][2])))
    
    #Joining query terms with idf from document
    query_tf_idf = query_words_tf.leftOuterJoin(idf_doc_words)
    
    #Checking common words and calculating the cosine similarity based on the formula
    query_tf_idf = query_tf_idf.map(lambda x : (x[0], 0 if x[1][1] is None  else  x[1][0]*x[1][1][1])).map(lambda x: ( (x[0], x[1])))
    query_doc_tf_idf = tf_idf_movies.map(lambda x: (x[1][0][0], (x[1][1], x[1][0][3]))).join(query_tf_idf).map(lambda x : (x[1][0][0], x[1][0][1] , x[1][1]) )
    query_doc_tf_idf = query_doc_tf_idf.map(lambda x: (x[0], (x[1]*x[2], x[2]*x[2], x[1]*x[1]))).reduceByKey(lambda x,y : ((x[0] + y[0], x[1] + y[1], x[2] + y[2])))
    
    #Sorting the documents in descenting order of their cosine similarity 
    result = query_doc_tf_idf.map(lambda x: (x[0], x[1][0]/(math.sqrt(x[1][1]) * math.sqrt(x[1][2]) ))).sortBy(keyfunc = lambda x: x[1] ,ascending=False)
    
  else:
    result = getMovieForTerm(query_terms[0])
  
  list_movie_names= result.map(lambda x:x[0]).take(10)
  print(list(list_movie_names))
  print("===================================================")
    


# COMMAND ----------


