import os
import sys
import copy
import time
import random
import pyspark
from statistics import mean
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import desc, size, max, abs

'''
INTRODUCTION

With this assignment you will get a practical hands-on of frequent 
itemsets and clustering algorithms in Spark. Before starting, you may 
want to review the following definitions and algorithms:
* Frequent itemsets: Market-basket model, association rules, confidence, interest.
* Clustering: kmeans clustering algorithm and its Spark implementation.

DATASET

We will use the dataset at 
https://archive.ics.uci.edu/ml/datasets/Plants, extracted from the USDA 
plant dataset. This dataset lists the plants found in US and Canadian 
states.

The dataset is available in data/plants.data, in CSV format. Every line 
in this file contains a tuple where the first element is the name of a 
plant, and the remaining elements are the states in which the plant is 
found. State abbreviations are in data/stateabbr.txt for your 
information.
'''

'''
HELPER FUNCTIONS

These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''


def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


def toCSVLineRDD(rdd):
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row])) \
        .reduce(lambda x, y: '\n'.join([x, y]))
    return a + '\n'


def toCSVLine(data):
    if isinstance(data, RDD):
        if data.count() > 0:
            return toCSVLineRDD(data)
        else:
            return ""
    elif isinstance(data, DataFrame):
        if data.count() > 0:
            return toCSVLineRDD(data.rdd)
        else:
            return ""
    return None


'''
PART 1: FREQUENT ITEMSETS

Here we will seek to identify association rules between states to 
associate them based on the plants that they contain. For instance, 
"[A, B] => C" will mean that "plants found in states A and B are likely 
to be found in state C". We adopt a market-basket model where the 
baskets are the plants and the items are the states. This example 
intentionally uses the market-basket model outside of its traditional 
scope to show how frequent itemset mining can be used in a variety of 
contexts.
'''


def data_frame(filename, n):
    '''
    Write a function that returns a CSV string representing the first
    <n> rows of a DataFrame with the following columns,
    ordered by increasing values of <id>:
    1. <id>: the id of the basket in the data file, i.e., its line number - 1 (ids start at 0).
    2. <plant>: the name of the plant associated to basket.
    3. <items>: the items (states) in the basket, ordered as in the data file.

    Return value: a CSV string. Using function toCSVLine on the right
                  DataFrame should return the correct answer.
    Test file: tests/test_data_frame.py
    '''
    spark = init_spark()
    sc = spark.sparkContext
    data = spark.read.text(filename).take(n)
    rdd = spark.sparkContext.parallelize(data)
    rdd = rdd.map(lambda x: x.value)
    data1 = sc.parallelize(range(0, n))
    res = rdd.map(lambda x: str(x).split(",")).map(lambda x: str(x[0]) + "," + str(x[1:]))
    res = data1.zip(res)
    return toCSVLineRDD(res)
    # return "not implemented"


def frequent_itemsets(filename, n, s, c):
    '''
    Using the FP-Growth algorithm from the ML library (see
    http://spark.apache.org/docs/latest/ml-frequent-pattern-mining.html),
    write a function that returns the first <n> frequent itemsets
    obtained using min support <s> and min confidence <c> (parameters
    of the FP-Growth model), sorted by (1) descending itemset size, and
    (2) descending frequency. The FP-Growth model should be applied to
    the DataFrame computed in the previous task.

    Return value: a CSV string. As before, using toCSVLine may help.
    Test: tests/test_frequent_items.py
    '''
    spark = init_spark()
    data = spark.read.text(filename).rdd
    rdd = data.map(lambda x: x.value)
    data1 = rdd.map(lambda x: Row(plant=(x.split(",")[0]), items=(x.split(",")[1:])))
    df = spark.createDataFrame(data1)
    fpgrowth = FPGrowth(itemsCol="items", minSupport=s, minConfidence=c)
    model = fpgrowth.fit(df)
    model = model.freqItemsets
    model = model.orderBy(size("items"), "freq", ascending=False).take(n)
    res = spark.sparkContext.parallelize(model)
    return toCSVLineRDD(res)
    # return "not implemented"


def association_rules(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that returns the
    first <n> association rules obtained using min support <s> and min
    confidence <c> (parameters of the FP-Growth model), sorted by (1)
    descending antecedent size in association rule, and (2) descending
    confidence.

    Return value: a CSV string.
    Test: tests/test_association_rules.py
    '''
    spark = init_spark()
    data = spark.read.text(filename).rdd
    rdd = data.map(lambda x: x.value)
    data1 = rdd.map(lambda x: Row(plant=(x.split(",")[0]), items=(x.split(",")[1:])))
    dframe = spark.createDataFrame(data1)
    fpgrowth = FPGrowth(itemsCol="items", minSupport=s, minConfidence=c)
    model = fpgrowth.fit(dframe)
    model = model.associationRules
    model = model.select("antecedent", "consequent", "confidence").orderBy(size("antecedent"), "confidence",
                                                                           ascending=False).take(n)
    res = spark.sparkContext.parallelize(model)
    return toCSVLineRDD(res)
    # return " not implemented"


def interests(filename, n, s, c):
    '''
    Using the same FP-Growth algorithm, write a script that computes
    the interest of association rules (interest = |confidence -
    frequency(consequent)|; note the absolute value)  obtained using
    min support <s> and min confidence <c> (parameters of the FP-Growth
    model), and prints the first <n> rules sorted by (1) descending
    antecedent size in association rule, and (2) descending interest.

    Return value: a CSV string.
    Test: tests/test_interests.py
    '''
    spark = init_spark()
    sc = spark.sparkContext
    data = spark.read.text(filename).rdd
    rdd = data.map(lambda x: x.value)
    data1 = rdd.map(lambda x: Row(plant=(x.split(",")[0]), items=(x.split(",")[1:])))
    dframe = spark.createDataFrame(data1)
    print("suhel")
    print(dframe)
    fpgrowth = FPGrowth(itemsCol="items", minSupport=s, minConfidence=c)
    model = fpgrowth.fit(dframe)
    assoc = model.associationRules
    freqitems = model.freqItemsets
    assoc = assoc.drop("lift")
    join = assoc.join(freqitems, assoc["consequent"] == freqitems["items"])

    new = join.withColumn("interest", abs(join["confidence"] - (join["freq"] / dframe.count())))
    new = new.drop("support")
    #print(new)
    final = new.orderBy(size("antecedent"), "interest", ascending=False).take(n)
    res = spark.sparkContext.parallelize(final)
    return toCSVLineRDD(res)
    # return " not implemented"


'''
PART 2: CLUSTERING

We will now cluster the states based on the plants that they contain.
We will reimplement and use the kmeans algorithm. States will be 
represented by a vector of binary components (0/1) of dimension D, 
where D is the number of plants in the data file. Coordinate i in a 
state vector will be 1 if and only if the ith plant in the dataset was 
found in the state (plants are ordered alphabetically, as in the 
dataset). For simplicity, we will initialize the kmeans algorithm 
randomly.

An example of clustering result can be visualized in states.png in this 
repository. This image was obtained with R's 'maps' package (Canadian 
provinces, Alaska and Hawaii couldn't be represented and a different 
seed than used in the tests was used). The classes seem to make sense 
from a geographical point of view!
'''

from all_states import all_states


def prep_data(items, plant):

    lst = []
    for i in range(len(all_states)):
         if all_states[i] in items:
             lst.append(1)
         else:
             lst.append(0)
    return plant, lst

def data_preparation(filename, plant, state):
    '''
    This function creates an RDD in which every element is a tuple with
    the state as first element and a dictionary representing a vector
    of plant as a second element:
    (name of the state, {dictionary})

    The dictionary should contains the plant names as keys. The
    corresponding values should be 1 if the plant occurs in the state
    represented by the tuple.

    You are strongly encouraged to use the RDD created here in the
    remainder of the assignment.

    Return value: True if the plant occurs in the state and False otherwise.
    Test: tests/test_data_preparation.py
    '''
    spark = init_spark()
    data = spark.read.text(filename).rdd
    rdd = data.map(lambda x: x.value)
    data1 = rdd.map(lambda x: Row(plant=(x.split(",")[0]), items=(x.split(",")[1:])))
    data2 = data1.map(lambda x: prep_data(x.items, x.plant))
    index = all_states.index(state)
    result = data2.filter(lambda x: x[0] == plant).map(lambda x: x[1][index]).collect()
    if result.pop() == 1:
        return True
    else:
        return False
    # return False


def distance2(filename, state1, state2):
    '''
    This function computes the squared Euclidean
    distance between two states.

    Return value: an integer.
    Test: tests/test_distance.py
    '''
    spark = init_spark()
    data = spark.read.text(filename).rdd
    rdd = data.map(lambda x: x.value)
    data1 = rdd.map(lambda x: Row(plant=(x.split(",")[0]), items=(x.split(",")[1:])))
    new = data1.map(lambda x: prep_data(x.items, x.plant))
    index1 = all_states.index(state1)
    index2 = all_states.index(state2)
    result = new.map(lambda x: ('a', (x[1][index1] - x[1][index2]) ** 2)).reduceByKey(lambda a, b: (a + b)).collect()
    return result[0][1]
    # return 42


def init_centroids(k, seed):
    '''
    This function randomly picks <k> states from the array in answers/all_states.py (you
    may import or copy this array to your code) using the random seed passed as
    argument and Python's 'random.sample' function.

    In the remainder, the centroids of the kmeans algorithm must be
    initialized using the method implemented here, perhaps using a line
    such as: `centroids = rdd.filter(lambda x: x[0] in
    init_states).collect()`, where 'rdd' is the RDD created in the data
    preparation task.

    Note that if your array of states has all the states, but not in the same
    order as the array in 'answers/all_states.py' you may fail the test case or
    have issues in the next questions.

    Return value: a list of <k> states.
    Test: tests/test_init_centroids.py
    '''
    random.seed(seed)
    return random.sample(all_states, k)
    # return []



def first_iter(filename, k, seed):
    '''
    This function assigns each state to its 'closest' class, where 'closest'
    means 'the class with the centroid closest to the tested state
    according to the distance defined in the distance function task'. Centroids
    must be initialized as in the previous task.

    Return value: a dictionary with <k> entries:
    - The key is a centroid.
    - The value is a list of states that are the closest to the centroid. The list should be alphabetically sorted.

    Test: tests/test_first_iter.py
    '''
    spark = init_spark()
    data = spark.read.text(filename).rdd
    rdd = data.map(lambda x: x.value)
    data1 = rdd.map(lambda x: Row(plant=(x.split(",")[0]), items=(x.split(",")[1:])))
    new = data1.map(lambda x: prep_data(x.items, x.plant))

    random.seed(seed)
    centroids = random.sample(all_states, k)
    result = {}
    for j in range(len(centroids)):
        result[centroids[j]] = []

    pd = new.map(lambda x: x[1]).toDF().toPandas()
    for i in range(len(all_states)):
        d = []
        for j in range(len(centroids)):
            index1 = all_states.index(all_states[i])
            index2 = all_states.index(centroids[j])
            pd[j] = ((pd['_' + str(index1 + 1)] - pd['_' + str(index2 + 1)]) ** 2)
            d.append(sum(pd[j]))
        tempcentroid = centroids[d.index(min(d))]
        lst = result[tempcentroid]
        lst.append(all_states[i])
        lst.sort()
        result[tempcentroid] = lst
    finalresult = {}
    for i in range(len(centroids)):
        finalresult[centroids[i]] = result[centroids[i]]
    return finalresult
    # return {}


def prepare_data(items, plant, centroids):
    lst = []
    for i in range(len(all_states)):
        if all_states[i] in items:
            lst.append(1)
        else:
            lst.append(0)

    for k in centroids:
        cnt = 0
        sum = 0
        for j in centroids[k]:
            id = all_states.index(j)
            cnt = cnt + 1
            sum = sum + lst[id]
        val = round((sum / cnt), 2)
        lst.append(val)

    return (plant, lst)


def kmeans(filename, k, seed):
    '''
    This function:
    1. Initializes <k> centroids.
    2. Assigns states to these centroids as in the previous task.
    3. Updates the centroids based on the assignments in 2.
    4. Goes to step 2 if the assignments have not changed since the previous iteration.
    5. Returns the <k> classes.

    Note: You should use the list of states provided in all_states.py to ensure that the same initialization is made.

    Return value: a list of lists where each sub-list contains all states (alphabetically sorted) of one class.
                  Example: [["qc", "on"], ["az", "ca"]] has two
                  classes: the first one contains the states "qc" and
                  "on", and the second one contains the states "az"
                  and "ca".
    Test file: tests/test_kmeans.py
    '''
    spark = init_spark()
    sc = spark.sparkContext
    data = spark.read.text(filename).rdd
    rdd = data.map(lambda x: x.value)
    dt = rdd.map(lambda x: Row(plant=(x.split(",")[0]), items=(x.split(",")[1:])))
    new = dt.map(lambda x: prep_data(x.items, x.plant))



    random.seed(seed)
    centroids = random.sample(all_states, k)
    result = {}
    for j in range(len(centroids)):
        result[centroids[j]] = []

    pd = new.map(lambda x: x[1]).toDF().toPandas()
    for i in range(len(all_states)):
        d = []
        for j in range(len(centroids)):
            index1 = all_states.index(all_states[i])
            index2 = all_states.index(centroids[j])
            pd[j] = ((pd['_' + str(index1 + 1)] - pd['_' + str(index2 + 1)]) ** 2)
            d.append(sum(pd[j]))
        tempcentroids = centroids[d.index(min(d))]
        lst = result[tempcentroids]
        lst.append(all_states[i])
        lst.sort()
        result[tempcentroids] = lst
    final = {}
    for i in range(len(centroids)):
        final[centroids[i]] = result[centroids[i]]



    new_states = all_states.copy()
    new_centroid = []
    a = 0
    for m in final:
        a = a + 1
        new_centroid.append(str(a))
        new_states.append(str(a))
    loop = 0



    while (True):
        new = dt.map(lambda x: prepare_data(x.items, x.plant, final))
        result = {}

        for j in range(len(new_centroid)):
            result[new_centroid[j]] = []

        pd = new.map(lambda x: x[1]).toDF().toPandas()
        pd1 = pd
        for i in range(len(all_states)):
            d = []
            pd = pd1
            for j in range(len(new_centroid)):
                index1 = new_states.index(all_states[i])
                index2 = new_states.index(new_centroid[j])
                pd[j] = ((pd['_' + str(index1 + 1)] - pd['_' + str(index2 + 1)]) ** 2)
                d.append(sum(pd[j]))
            tempcentroids= new_centroid[d.index(min(d))]
            lst = result[tempcentroids]
            lst.append(new_states[i])
            lst.sort()
            result[tempcentroids] = lst
        new_final = {}
        for i in range(len(new_centroid)):
            new_final[new_centroid[i]] = result[new_centroid[i]]
        loop = loop + 1
        if final == new_final:
            break
        if loop == 3:
            break
        final = {}
        final = new_final.copy()

    output = []
    for k in final:
        output.append(final[k])

    return output
    return []