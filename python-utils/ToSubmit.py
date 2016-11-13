#!/usr/bin/python
import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import  roc_auc_score
from sklearn.metrics import matthews_corrcoef
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import sys

import matplotlib.pyplot as plt
import numpy as np
import os; import glob
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
    mccs = np.zeros(n)
    for i in range(n):
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc



conf = SparkConf()
conf.setMaster("local[4]")


spark = SparkSession \
    .builder.config(conf=conf) \
    .appName("bosch-spark for submission").getOrCreate()


# Read in the Parquet file created above. Parquet files are self-describing so the schema is preserved.
# The result of loading a parquet file is also a DataFrame.
directory="../results"
latest=max(glob.glob(os.path.join(directory, '*/')), key=os.path.getmtime)
print latest
comp = spark.read.parquet(latest+"preds.parquet").toPandas()

print comp.head(10)
preds0=comp["probabilities"]
#print len(preds0)
response=comp["label"]

preds1=np.empty(len(preds0))
for i in range(len(preds0)):
    preds1[i]=preds0[i][1]


auc=roc_auc_score(np.asarray(response),np.asarray(preds1))
print ("auc=%g" %auc)

# pick the best threshold out-of-fold
best_threshold0, best_mcc, predsGBT=eval_mcc(np.asarray(response),np.asarray(preds1), show=True)
preds=np.asarray(predsGBT).astype(np.int16)
response=np.asarray(response).astype(np.int16)
score=matthews_corrcoef(response,preds)
print ("mcc=%g" %score)
print ('best_threshold0=%g' %best_threshold0)

# pick the best threshold out-of-fold
thresholds = np.linspace(0.1, 0.5, 500)
mcc_score = np.array([matthews_corrcoef(response, preds1>thr) for thr in thresholds])
best_threshold = thresholds[mcc_score.argmax()]
print ('best_threshold=%g' %best_threshold)
print("mbest_mcc=%s" %mcc_score.max())
 
# Read probabilitiess.
probabilitiess = spark.read.parquet(latest+"submission.parquet").toPandas()
print probabilitiess.head(10)
idvalue=probabilitiess["id"]
preds0=probabilitiess["probabilities"]
preds1=np.empty(len(preds0))
for i in range(len(preds0)):
    preds1[i]=preds0[i][1]
thresh=(best_threshold+best_threshold0)/2.0
predsGBT = ( preds1> thresh)
print ("thresh=%g" %thresh)
print ("mcc=%g" %score)
print len(predsGBT)
sub = pd.read_csv('../input/sample_submission.csv')
sub['Response'] = np.asarray(predsGBT).astype(np.int16)
sub['Id'] = np.asarray(idvalue)
sub.to_csv('../results/bosch-spark-xgboost.csv.gz', index=False,compression="gzip")
