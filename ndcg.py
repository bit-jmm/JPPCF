#!/usr/bin/python

"""
  This script is used to get NDCG@k with file containing records of

  ranked ground turth to queries and k as input. See paper "Gradient descent
  optimization of smoothed information retrieval metrics" for more details
"""


import math
import copy


# This is a function to get maxium value of DCG@k.
# That is the DCG@k of sorted ground truth list.
def get_max_ndcg(k, *ins):
    l = [i for i in ins]
    l = copy.copy(l[0])
    l.sort(None, None, True)
    max_num = 0.0
    for i in range(k):
        max_num += (math.pow(2, l[i]) - 1) / math.log(i + 2, 2)
    return max_num


# This is a function to get ndcg
def get_ndcg(s, k):
    z = get_max_ndcg(k, s)
    dcg = 0.0
    for i in range(k):
        dcg += (math.pow(2, s[i]) - 1) / math.log(i + 2, 2)
    if z == 0:
        z = 1
    ndcg = dcg / z
    return ndcg

