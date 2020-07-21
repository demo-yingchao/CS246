import os

# Set spark environments
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

data = sc.textFile('q2/data/graph-full.txt')
data = data.map(lambda x: x.split('\t')).map(lambda x: (int(x[0]), int(x[1])))
data = data.distinct()
deg = data.groupByKey().mapValues(len)
data = data.leftOuterJoin(deg)
data.persist()

n = 1000
iterations = 40
beta = 0.8
res = (1-beta)/n

r = [1/n]*n
r_bc = sc.broadcast(r)

zero_value = [0]*n

def seq_op(r_, item):
  s, (t, deg) = item
  r_[t-1] += r_bc.value[s-1]/deg
  return r_

def comb_op(r1, r2):
  for i in range(n):
    r1[i] += r2[i]
  return r1

for _ in range(iterations):
  zero_value = [0]*n
  r = data.aggregate(zero_value, seq_op, comb_op)
  for i in range(n):
    r[i] = beta*r[i] + res
  print(np.sum(np.array(r)))
  r_bc = sc.broadcast(r)

print(np.argmax(np.array(r)))
print(np.max(np.array(r)))
  
