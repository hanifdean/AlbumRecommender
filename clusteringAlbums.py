# Databricks notebook source
# MAGIC %md
# MAGIC ## Import

# COMMAND ----------

from numpy import array
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
import numpy as np
import pandas as pd
import math

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable Initialization

# COMMAND ----------
conf = SparkConf().setMaster('spark://ec2-52-23-167-89.compute-1.amazonaws.com:7077')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)
FEAT_NUM = 13
FIXED_TRACK_NUM = 24
FIRST_FILE = '/home/ec2-user/AlbumRecommender/dataset-7k-2010.txt'
FILENAMES = ['/home/ec2-user/AlbumRecommender/dataset-7k-2011.txt',
             '/home/ec2-user/AlbumRecommender/dataset-7k-2012.txt',
             '/home/ec2-user/AlbumRecommender/dataset-7k-2013.txt',
             '/home/ec2-user/AlbumRecommender/dataset-7k-2014.txt',
             '/home/ec2-user/AlbumRecommender/dataset-7k-2015.txt',
             '/home/ec2-user/AlbumRecommender/dataset-7k-2016.txt',
             '/home/ec2-user/AlbumRecommender/dataset-7k-2017.txt',
             '/home/ec2-user/AlbumRecommender/dataset-7k-2018.txt'
            ]
SAVE_ALBUM_CLUSTER_FILE = 's3a://album-recommender-bucket/withAlbumAsKey'
SAVE_CLUSTER_ALBUM_FILE = 's3a://album-recommender-bucket/withClusterAsKey'
albumSchema = StructType([ \
    StructField("AI", StringType()), \
    StructField("AN", StringType()), \
    StructField("TI", StringType()), \
    StructField("TN", StringType()), \
    StructField("FTS", ArrayType(DoubleType()))])

# COMMAND ----------

def loadFileAsList(path):
  albums = []
  albumFile = open(path, 'r')
  for line in albumFile:
    albums.append(line.rstrip().split('\t'))
    for x in range(len(albums[-1])):
      if x > 3:
        albums[-1][x] = float(albums[-1][x])
    albums[-1] = [albums[-1][0], albums[-1][1], albums[-1][2], albums[-1][3], [f for f in albums[-1][4:]]]
  albumFile.close()
  return albums

# COMMAND ----------

def interpolateRow(row):
  finalRow = [row.AN_AI, [ [ np.nan for _ in range(FEAT_NUM) ] for _ in range(FIXED_TRACK_NUM) ]]

  # For each feature type
  for i in range(FEAT_NUM):

    tempFeatures = []

    # Divide case to albums with single track and multiple tracks
    # to avoid 'Division by zero' exception
    # If multiple tracks
    if(len(row.FTS) != 1):
      interSpace = int(math.floor(((FIXED_TRACK_NUM - len(row.FTS)) / (len(row.FTS) - 1))))
      print interSpace
      additionalSpace = (FIXED_TRACK_NUM - len(row.FTS)) % (len(row.FTS) - 1)

      ## Value and NaN placements
      # For each list of track features
      for j in row.FTS:
        tempFeatures.append(j[i])
        for s in range(interSpace):
          tempFeatures.append(np.nan)
        if additionalSpace > 0:
          tempFeatures.append(np.nan)
          additionalSpace -= 1

    # If single track
    else:
      tempFeatures.append(row.FTS[0][i])
      for n in range(FIXED_TRACK_NUM - 1):
        tempFeatures.append(np.nan)

    ## Interpolate
    tempFeatures = pd.Series(tempFeatures)
    tempFeatures = tempFeatures.interpolate()

    ## Assign Back
    for j in range(FIXED_TRACK_NUM):
      finalRow[1][j][i] = tempFeatures[j]

  return finalRow

# COMMAND ----------

def stringify(data):
  row = data[0]
  for x in data[1]:
    for y in x:
      row += '\t%f' % y
  return row

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the interpolated album RDD

# COMMAND ----------

finalAlbums = (sqlContext
               .createDataFrame(loadFileAsList(FIRST_FILE), albumSchema)
               .withColumn("AN_AI",concat(col("AN"), lit("_"), col("AI")))
               .groupBy("AN_AI")
               .agg(collect_list('FTS').alias('FTS'))
               .filter(size('FTS') <= FIXED_TRACK_NUM)
               .rdd.map(interpolateRow)
               .cache())

for fn in FILENAMES:
  df = (sqlContext
               .createDataFrame(loadFileAsList(fn), albumSchema)
               .withColumn("AN_AI",concat(col("AN"), lit("_"), col("AI")))
               .groupBy("AN_AI")
               .agg(collect_list('FTS').alias('FTS'))
               .filter(size('FTS') <= FIXED_TRACK_NUM)
               .rdd.map(interpolateRow))
  finalAlbums = finalAlbums.union(df).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the difference between data points

# COMMAND ----------

# print finalAlbums.take(1)

# COMMAND ----------

def getDiff(row):
  row_train_data = []
  for i in range(FIXED_TRACK_NUM-1):
    for j in range(FEAT_NUM):
      row_train_data.append(row[1][i+1][j]-row[1][i][j])
  return [row[0], Vectors.dense(row_train_data)]

# COMMAND ----------

# print len(finalAlbums.map(getDiff).take(1)[0][1])
# print 23*13

# COMMAND ----------

df = sqlContext.createDataFrame(finalAlbums.map(getDiff), ['id', 'features'])

# COMMAND ----------

# df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run through KMeans

# COMMAND ----------

k_list = [500, 1000, 3000, 7000]
best_model = 0
best_score = -99
final_df = 0
evaluator = ClusteringEvaluator()

for c in k_list:
  kmeans = KMeans()
  curr_model = kmeans.fit(df)
  curr_res = curr_model.transform(df)
  curr_score = evaluator.evaluate(curr_res)
  if curr_score > best_score:
      best_model = curr_model
      best_score = curr_score
      final_df = curr_res


# COMMAND ----------

# MAGIC %md
# MAGIC ## save results into two text files

# COMMAND ----------

def stringifyList(data):
  row = str(data[0])
  for x in data[1]:
    row += '\t%s' % x
  return row

# COMMAND ----------

def stringify(data):
  row = data[0] + '\t%d' % data[1]
  return row

# COMMAND ----------

transformed = final_df.withColumnRenamed('prediction','cluster_num')

# COMMAND ----------

transformed.drop('features').rdd.map(stringify).coalesce(1).saveAsTextFile(SAVE_ALBUM_CLUSTER_FILE)

# COMMAND ----------

transformed.groupBy('cluster_num').agg(collect_list('id').alias('albums_list')).rdd.map(stringifyList).coalesce(1).saveAsTextFile(SAVE_CLUSTER_ALBUM_FILE)

# COMMAND ----------
