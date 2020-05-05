#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pyspark
import argparse
from pyspark.ml.regression import GBTRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import PipelineModel, Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler                    
from pyspark.ml.regression import GBTRegressor
import pyspark.sql.functions as func
from pyspark.mllib.evaluation import MulticlassMetrics


# In[2]:


spark = SparkSession.builder.appName('WineApp_Prediction').getOrCreate()


# In[44]:


def read_csv(file_path):
    return spark.read.format("com.databricks.spark.csv").csv(
        file_path, header=True, sep=";")


# In[45]:


def load_model(path, classifier='gbt'):
    if classifier=='gbt':
        return GBTRegressionModel.load(path) # for gbt
    return Pipeline.load(path) # for random forest


# In[58]:


def preprocess(df, classifier='gbt'):
    total_columns = df.columns
    df = df.select(*(col(c).cast("double").alias(c) for c in df.columns))

    stages = []
    unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

    for column_name in total_columns[:-1]:
        stages = []
        vectorAssembler = VectorAssembler(inputCols=[column_name],outputCol=column_name+'_vect')
        stages.append(vectorAssembler)
        stages.append(MinMaxScaler(inputCol=column_name+'_vect', outputCol=column_name+'_scaled'))
        pipeline = Pipeline(stages=stages)
        df = pipeline.fit(df).transform(df).withColumn(
            column_name+"_scaled", unlist(column_name+"_scaled")).drop(
            column_name+"_vect").drop(column_name)
    # if you want to run gbt, don't comment next 2 lines
    if classifier == 'gbt':
        vectorAssembler = VectorAssembler(
            inputCols=[column_name+'_scaled' for column_name in total_columns[:-1]],
            outputCol='features')
        df = vectorAssembler.transform(df)
    return df, total_columns


# In[59]:


def get_predictions(model, df):
    return model.transform(df)


# In[60]:


def run(test_file, classifier='gbt'):
    df = read_csv(test_file)
    df, total_columns = preprocess(df)
    if classifier == 'gbt':
        model = load_model("s3a://wineappcloud/gbt.model")
    elif classifier == 'rf':
        model = load_model("s3a://wineappcloud/RandomForestClassifier.model")
    else:
        model = load_model("s3a://wineappcloud/DecisionTreeClassifier.model")
    df = get_predictions(model, df)
    if classifier =='gbt':
        df = df.withColumn("prediction_with_round", func.round(df["prediction"], 0)).drop('prediction')
        df = df.select("prediction_with_round", total_columns[-1])
    return df, total_columns


# In[61]:


def print_f1(df, total_columns, classifier='gbt'):
    label_column = total_columns[-1]
    if classifier == 'gbt':
        predictionAndLabels = df.select(['prediction_with_round', total_columns[-1]])
    else:
        predictionAndLabels = df.select(['indexedLabel', "prediction"])
    labels = df.select([label_column]).distinct()
    header = labels.rdd.first()
    labels = labels.rdd.filter(lambda line: line !=header)
    header = predictionAndLabels.rdd.first()
    copy_predictionAndLabels = predictionAndLabels.rdd.filter(lambda line: line != header)
    copy_predictionAndLabel = copy_predictionAndLabels.map(lambda lp: (float(lp[0]), float(lp[1])))
    metrics = MulticlassMetrics(copy_predictionAndLabel)
    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)


# In[62]:

import argparse
parser = argparse.ArgumentParser(description='Wine Quality prediction')
parser.add_argument('--test_file', required=True, help='please provide test file path you can provide s3 path or local file path')
args = parser.parse_args()
df, total_columns = run(args.test_file)
print_f1(df, total_columns)


# In[ ]:




