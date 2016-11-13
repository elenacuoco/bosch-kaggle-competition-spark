"""
@author Elena Cuoco
@email: info@elenacuoco.com
"""
 
import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef as mcc
import pyspark
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession

from pyspark.sql import Row
from pyspark.sql.functions import UserDefinedFunction
from pyspark.ml.linalg import DenseVector
from pyspark.sql.types import *
import atexit
from numpy import array
import numpy as np
import datetime
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import PCA, MinMaxScaler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
 
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
import gc
from pyspark.sql.functions import col, count, sum
print ("Successfully imported Spark Modules")
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes,MultilayerPerceptronClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.metrics import matthews_corrcoef
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit
from pyspark.sql.functions import rand
# from numba import jit

conf = SparkConf()
conf.set("spark.executor.memory", "6G")
conf.set("spark.driver.memory", "4G")
conf.set("spark.executor.cores", "4")
conf.set("spark.sql.crossJoin.enabled", "true")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.default.parallelism", "4")
conf.setMaster('local[4]')

atexit.register(lambda: spark.stop())

spark = SparkSession \
    .builder.config(conf=conf) \
    .appName("bosch-spark-magic").getOrCreate()


# @jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)


def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)

        return best_proba, best_mcc, y_pred
    else:
        return best_mcc


# spark is an existing SparkSession
df0 = spark.read.csv("../input/train_numeric.csv", header="true", inferSchema="true")
df_test0 = spark.read.csv("../input/test_numeric.csv", header="true", inferSchema="true")
 
leak = spark.read.csv("../input/happy.csv", header="true", inferSchema="true")
magic = spark.read.csv("../input/magic.csv", header="true", inferSchema="true")
df00 = df0.join(leak, df0.Id == leak.Id).drop(df0.Id)
df_test00 = df_test0.join(leak, df_test0.Id == leak.Id).drop(df_test0.Id)
df0 = df00.join(magic, df00.Id == magic.Id).drop(df00.Id)
df_test0 = df_test00.join(magic, df_test00.Id == magic.Id).drop(df_test00.Id)

dateset = spark.read.csv("../input/train_date.csv", header="true", inferSchema="true")
datetest = spark.read.csv("../input/test_date.csv", header="true", inferSchema="true")

dateset=dateset.na.fill(9999999)
datetest=datetest.na.fill(9999999)
ignore = ['Id', 'L3_S46_D4135']
lista = [x for x in dateset.columns if x not in ignore]
assembler = VectorAssembler(
    inputCols=lista,
    outputCol='IDatefeatures')
dateData = (assembler.transform(dateset).select("Id","IDatefeatures"))
dateDatatest = (assembler.transform(datetest).select("Id","IDatefeatures"))
print("VectorAssembler Date Done!")
selector = PCA(inputCol="IDatefeatures",outputCol="Datefeatures",k=150)
selectorModel = selector.fit(dateData)
dateSel = selectorModel.transform(dateData).select("Id", "Datefeatures")
dateSeltest = selectorModel.transform(dateDatatest).select("Id", "Datefeatures")
print("selector for Date Done!")
df= df0.join(dateSel, df0.Id == dateSel.Id).drop(df0.Id)
df_test = df_test0.join(dateSeltest, df_test0.Id == dateSeltest.Id).drop(df_test0.Id)
del dateSeltest,dateSel, dateData,dateDatatest,dateset,datetest
gc.collect()

df = df.na.fill(9999999)
df_test = df_test.na.fill(9999999)

# df.printSchema()


ignore = ['Id', 'Response']
lista = [x for x in df.columns if x not in ignore]
assembler = VectorAssembler(
    inputCols=lista,
    outputCol='features')
data = (assembler.transform(df).select("features", df.Response.astype('double')))
data_test = (assembler.transform(df_test).select('Id',"features"))
data.printSchema()
 
# Split the data into training and test sets (20% held out for testing)
(trainingData, testData) = data.randomSplit([0.8, 0.2], seed=451)

# Train a Random Forest model.

cls = RandomForestClassifier(numTrees=600, seed=1111, maxDepth=15, labelCol="Response", featuresCol="features")

 
pipeline = Pipeline(stages=[cls])
evaluator = MulticlassClassificationEvaluator(
    labelCol="Response", predictionCol="prediction", metricName="accuracy")

model = pipeline.fit(trainingData)
predictions = model.transform(testData)
predictions.select("probability", "prediction", "Response", "features").show(5)

# Select (prediction, true label) and compute test error

accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

preds1 = predictions.select("probability").rdd.map(lambda r: r[0][1]).collect()
response = predictions.select("Response").rdd.map(lambda r: r[0]).collect()

# use https://www.kaggle.com/cpmpml best_threshold funtion for mcc
best_threshold, best_mcc, predsGBT = eval_mcc(np.asarray(response), np.asarray(preds1), show=True)
print ("best_threshold =%g" % best_threshold)
 
preds = np.asarray(predsGBT).astype(int)
response = np.asarray(response).astype(int)
score = matthews_corrcoef(response, preds)
print ("mcc=%g" % score)

# Make predictions.
predictions = model.transform(data_test)
sub = pd.read_csv('../input/sample_submission.csv')
preds1 = predictions.select("probability").rdd.map(lambda r: r[0][1]).collect()
tosave=pd.DataFrame()
tosave['Id']=sub['Id']
tosave['probability']=preds1
tosave.to_csv('../output/rf-magic.csv',index=False)

predsGBT = (preds1 > best_threshold)
 

sub['Response'] = np.asarray(predsGBT).astype(int)
sub.to_csv('../output/bosch-spark-magic-dec1.csv.gz', index=False, compression="gzip")
spark.stop()
