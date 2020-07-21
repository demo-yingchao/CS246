import os

# Set spark environments
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

text = spark.read.text('hw1-bundle/q1/data/soc-LiveJournal1Adj.txt')
t1 = text.withColumn("value", split(col('value'), '\t'))
data = t1.select(
    col('value').getItem(0).alias('v').cast('int'),
    col('value').getItem(1).alias('d1')).drop('value')
data = data.withColumn('d1', split(col('d1'), ',').cast('array<int>'))

def get_d2(s):
  l = s['d1']
  ret = []
  for i,v in enumerate(l):
    ret.append((v, l[0:i]+l[i+1:]))
  return ret

def seq_op(ct, l):
  if l:
    for v in l:
      if v not in ct:
        ct[v] = 1
      else:
        ct[v] += 1
  return ct

def comb_op(ct1, ct2):
  if(len(ct2) > len(ct1)):
    t = ct2
    ct2 = ct1
    ct1 = t
  for v in ct2:
    if v not in ct1:
      ct1[v] = 1
    else:
      ct1[v] += 1
  return ct1
    
d1 = data.select('d1').rdd.flatMap(get_d2)
d2 = d1.aggregateByKey({}, seq_op, comb_op)
d2 = spark.createDataFrame(d2, ['v', 'd2'])

data = data.join(d2, 'v', 'left')

def func(col1, col2):
  if not col1:
    return []
  d = {}
  for k,v in col1.items():
    if not k in col2:
      d[k] = v
  ret = sorted(d.items(), key=lambda x:(-x[1], x[0]))[0:10]
  return [i[0] for i in ret]

subsort = udf(func, ArrayType(IntegerType()))
_ = spark.udf.register('subsort', subsort)

d3 = data.withColumn('rec', subsort('d2', 'd1')).drop('d1', 'd2')

print(d3[d3.v==11].collect())

