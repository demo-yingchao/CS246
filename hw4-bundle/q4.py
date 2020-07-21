import os

# Set spark environments
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from math import log

import pyspark
from pyspark.sql.session import SparkSession
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

def create_hash_function(a, b, p, n_buckets):
  def hash_function(x):
    y = x%p
    hash_val = (a*y+b)%p
    return hash_val % n_buckets
  return hash_function

func = []
p = 123457
n_buckets = 10000
with open('q4/data/hash_params.txt', 'r') as f:
  lines = f.readlines()
  for l in lines:
    a, b = [int(x) for x in l[:-1].split('\t')]
    func.append(create_hash_function(a, b, p, n_buckets))
    
rf = []
with open('q4/data/counts.txt', 'r') as f:
  rf = f.readlines()
  rf = [int(l[:-1].split('\t')[-1]) for l in rf]

zero_value = [[0]*n_buckets]*5

def seq_op(v, item):
  for i in range(len(v)):
    hash_val = func[i](item)
    v[i][hash_val] += 1
  return v

def comb_op(v1, v2):
  for i in range(len(v1)):
    for j in range(n_buckets):
      v1[i][j] += v2[i][j]
  return v1


data = sc.textFile('q4/data/words_stream.txt')
T = data.count()
data = data.map(lambda x: int(x))
f = data.aggregate(zero_value, seq_op, comb_op)
#f = np.min(f, 0)

np.save('f.npy', f, allow_pickle=True)

f = np.load('f.npy', allow_pickle=True)

length = len(rf)
err = [0]*length
for i in range(length):
  hv = [0]*5
  for p in range(5):
    hv[p] = f[p][func[p](i+1)]
#hmin = reduce(lambda x,y: min(x,y), hv)
  hmin = reduce(lambda x,y: x if x <=y else y, hv)
  err[i] = (hmin - rf[i])/rf[i]

x = [log(k/T, 10) for k in rf]
y = [log(e, 10) for e in err]

plt.figure()
plt.plot(x, y, '.')
plt.show()
