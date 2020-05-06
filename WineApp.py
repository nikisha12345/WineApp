#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyspark


# In[ ]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler                    
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# In[ ]:


# sc.install_pypi_package("scikit-learn")
# from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


spark = SparkSession.builder.appName('WineApp').getOrCreate()


# In[ ]:


train_dataset= spark.read.format("com.databricks.spark.csv").csv(
    's3a://wineappcloud/TrainingDataset.csv', header=True, sep=";")
train_dataset.printSchema()


# In[ ]:


validation_dataset= spark.read.format("com.databricks.spark.csv").csv(
    's3a://wineappcloud/ValidationDataset.csv', header=True, sep=";")
validation_dataset.printSchema()


# In[ ]:


train_dataset.show()


# In[ ]:



validation_dataset.show()


# In[ ]:


train_dataset=train_dataset.distinct()
validation_dataset=validation_dataset.distinct()


# In[ ]:


train_dataset.count()


# In[ ]:


validation_dataset.count()


# In[ ]:


total_columns = train_dataset.columns
tot_columns=validation_dataset.columns


# In[ ]:


train_dataset.show()


# In[ ]:


from pyspark.sql.functions import col
def preprocess(dataset):
    return dataset.select(*(col(c).cast("double").alias(c) for c in dataset.columns))
train_dataset = preprocess(train_dataset)
validation_dataset = preprocess(validation_dataset)


# In[ ]:


train_dataset.show()


# In[ ]:


validation_dataset.show()


# # Minmax

# In[ ]:


from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
stages = []
unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

for column_name in total_columns[:-1]:
    stages = []
    vectorAssembler = VectorAssembler(inputCols=[column_name],outputCol=column_name+'_vect')
    stages.append(vectorAssembler)
    stages.append(MinMaxScaler(inputCol=column_name+'_vect', outputCol=column_name+'_scaled'))
    pipeline = Pipeline(stages=stages)
    train_dataset = pipeline.fit(train_dataset).transform(train_dataset).withColumn(
        column_name+"_scaled", unlist(column_name+"_scaled")).drop(column_name+"_vect").drop(column_name)


# In[ ]:


train_dataset.show(5)


# In[ ]:


for column_name in total_columns[:-1]:
    stages = []
    vectorAssembler = VectorAssembler(inputCols=[column_name],outputCol=column_name+'_vect')
    stages.append(vectorAssembler)
    stages.append(MinMaxScaler(inputCol=column_name+'_vect', outputCol=column_name+'_scaled'))
    pipeline = Pipeline(stages=stages)
    validation_dataset = pipeline.fit(validation_dataset).transform(validation_dataset).withColumn(
        column_name+"_scaled", unlist(column_name+"_scaled")).drop(column_name+"_vect").drop(column_name)


# In[ ]:


validation_dataset.show(5)


# In[ ]:


vectorAssembler = VectorAssembler(
    inputCols=[column_name+"_scaled" for column_name in total_columns[:-1]],
    outputCol='features')
train_dataset.columns
wine_train = vectorAssembler.transform(train_dataset)
wine_valid = vectorAssembler.transform(validation_dataset)
wine_data_train = wine_train.select(['features',total_columns[-1]]).cache()
wine_data_valid = wine_valid.select(['features',total_columns[-1]]).cache()


# In[ ]:


wine_data_train.show()


# In[ ]:


wine_data_valid.show()


# In[ ]:


gbt = GBTRegressor(featuresCol='features',
                    labelCol = total_columns[-1],
                    maxIter=100,
                    maxDepth=5,
                    subsamplingRate=0.5,
                    stepSize=0.1)
gbt_model = gbt.fit(wine_data_train)
gbt_predictions = gbt_model.transform(wine_data_valid)
gbt_predictions.select('prediction',total_columns[-1]).show(5)
# gbt_model.save("gbt.model")

gbt_model.write().overwrite().save("s3a://wineappcloud/gbt.model")

gbt_evaluator1= RegressionEvaluator(labelCol = total_columns[-1],
                                   predictionCol='prediction',
                                   metricName="rmse")
rmse = gbt_evaluator1.evaluate(gbt_predictions)


gbt_evaluator2= RegressionEvaluator(labelCol = total_columns[-1],
                                   predictionCol='prediction',
                                   metricName="r2")
r2 = gbt_evaluator2.evaluate(gbt_predictions)


print("RMS=%g" % rmse)
print("R squared = ",r2)


import pyspark.sql.functions as func

gbt_predictions = gbt_predictions.withColumn("prediction_with_round", func.round(gbt_predictions["prediction"], 0))
gbt_predictions.select(['prediction_with_round']).show(5)
y_true = wine_data_valid.select([total_columns[-1]]).collect()
y_pred = gbt_predictions.select(['prediction_with_round']).collect()

# # RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

labelIndexer = StringIndexer(inputCol=total_columns[-1], outputCol="indexedLabel").fit(train_dataset)


rf = RandomForestClassifier(labelCol='indexedLabel', featuresCol="features", numTrees=100)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


pipeline = Pipeline(stages=[labelIndexer, vectorAssembler, rf, labelConverter])


model = pipeline.fit(train_dataset)
model.write().overwrite().save("s3a://wineappcloud/RandomForestClassifier.model")
# Make predictions.
predictions = model.transform(validation_dataset)

# Select example rows to display.
predictions.select("predictedLabel", total_columns[-1], "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only


print(accuracy)


# # DecisionTreeClassifier
from pyspark.ml.classification import DecisionTreeClassifier
bt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")
pipeline = Pipeline(stages=[labelIndexer, vectorAssembler, bt, labelConverter])
model = pipeline.fit(train_dataset)
model.write().overwrite().save("s3a://wineappcloud/DecisionTreeClassifier.model")
# Make predictions.
predictions = model.transform(validation_dataset)

# Select example rows to display.
predictions.select("predictedLabel", total_columns[-1], "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

print(accuracy)

