
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

data = sc.textFile('q2/data/graph-small.txt')
data = data.map(lambda x: x.split('\t')).map(lambda x: (int(x[0]), int(x[1])))
data = data.distinct()
data.persist()

n = 100
iterations = 40



def seq_op2(r, item):
  s, t = item
  r[t-1] += bc.value[s-1]
  return r

def seq_op1(r, item):
  s, t = item
  r[s-1] += bc.value[t-1]
  return r

def comb_op(r1, r2):
  for i in range(n):
    r1[i] += r2[i]
  return r1

h = [1]*n
bc = sc.broadcast(h)

for _ in range(iterations):
  zero_value = [0]*n
  a = data.aggregate(zero_value, seq_op1, comb_op)
  ma = np.max(a)
  for i in range(n):
    a[i] = a[i]/ma
  bc = sc.broadcast(a)

  zero_value = [0]*n
  h = data.aggregate(zero_value, seq_op2, comb_op)
  mh = np.max(h)
  for i in range(n):
    h[i] = h[i]/mh
  bc = sc.broadcast(h)


print('max hub id')
print(np.argmax(np.array(h)))
print('max auth id')
print(np.argmax(np.array(a)))
  
