import os

# Set spark environments
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

text = spark.read.text('q2/data/browsing.txt')
data = text.withColumn('value', split(col('value'), ' '))
data = data.rdd.map(lambda x: frozenset([s for s in x[0] if s])).collect()
D = sc.broadcast(data)

minsup = 100
L1 = sc.parallelize(data).flatMap(lambda x: [(s, 1) for s in x]).reduceByKey(lambda a, b: a+b).filter(lambda x: x[1] >= minsup).collect()
L1 = [(frozenset({s[0]}), s[1]) for s in L1]
print(len(L1))
print(len(data))

tmp = L1[0]
print('-------------')

C1 = [x[0] for x in L1]

#tuple (item, count)
L = [L1]
# set {item}
C = [C1]

M = 3
for i in range(M-1):
  item = i+2
  prev = C[i]
  cand = [v1|v2 for p, v1 in enumerate(prev) for v2 in prev[p+1:] if len(v1|v2) == item]
  if i > 0:
    side = item*(item-1)//2
    cand = sc.parallelize(cand).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a+b).filter(lambda x: x[1] == side).keys().collect()
  L_new = sc.parallelize(cand).map(lambda x: (x, len([0 for s in D.value if x.issubset(s)]))).filter(lambda x: x[1] >= minsup).collect()
  L.append(L_new)
  C_new = [x[0] for x in L_new]
  C.append(C_new)


with open('apriori-L', 'wb') as f:
  pickle.dump(L, f)
  print('pickle dump succeed')

L = 0
with open('apriori-L', 'rb') as f:
  L = pickle.load(f)
  
M = 3  
for i in range(M-1):
  rule = []
  for freqset, val in L[i+1]:
    L_d = dict(L[i])
    for _conset in freqset:
      conset = frozenset({_conset})
      conf_val = val/L_d[freqset - conset]
      rule.append((freqset-conset, conset, conf_val))
  rule = sorted(rule, key=lambda x: x[2], reverse=True)
  for r in rule[0:5]:
    print('%s-->%s, confidence: %f' %(r[0], r[1], r[2]))
  print('-----------------------')
      
