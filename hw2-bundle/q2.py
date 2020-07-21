import os

# Set spark environments
os.environ['PYSPARK_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3'
os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'

'''current problem is that grobals in udf nerver changed
   and every r seems occur twice?'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import copy

import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

data = spark.read.text('q2/data/data.txt')
data = data.withColumn('value', split('value', ' ').cast('array<double>')) 

cent = spark.read.text('q2/data/c1.txt')
cent = cent.withColumn('value', split('value', ' ').cast('array<double>')).rdd.map(lambda r: r[0]).collect()

C_bc = sc.broadcast(np.array(cent))

def AssignCenter(r):
  ret = np.sum((np.array(C_bc) - r)**2, 1)
  c_index = ret.argmin()
  c_dist = ret[c_index]
  return int(c_index),float(c_dist)

assign_center = udf(AssignCenter, StructType([
      StructField('center', IntegerType(), False),
      StructField('dist', DoubleType(), False)]))

maxiter = 2

@pandas_udf(ArrayType(DoubleType()), PandasUDFType.GROUPED_AGG)
def array_avg(v):
  return np.mean(v, 0)
_ = spark.udf.register('array_avg', array_avg)

@pandas_udf(DoubleType(), PandasUDFType.GROUPED_AGG)
def double_avg(v):
  return np.sum(v)
_ = spark.udf.register('double_avg', double_avg)

y_cost = []
for i in range(maxiter):
  udf_data = data.withColumn('center_cost', assign_center('value'))
  assign_data = udf_data.withColumn('center', col('center_cost').getItem('center').alias('center')).withColumn('cost', col('center_cost').getItem('dist').alias('cost')).drop('center_cost')
  gdata = assign_data.groupBy('center').agg({'value':'array_avg', 'cost':'double_avg'}).collect()
  new_center = [r['array_avg(value)'] for r in gdata]
  C_bc = sc.broadcast(np.array(new_center))
  
  cost = reduce(lambda x,y: x+y, [r['double_avg(cost)'] for r in gdata])
  if i == 0:
    cost0 = cost
    y_cost.append(1)
  else:
    y_cost.append(cost/cost0)

plt.figure('c1 initialize')
plt.plot(y_cost, '-.^')
plt.show()
    
  
