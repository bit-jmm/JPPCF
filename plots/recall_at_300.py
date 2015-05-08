import numpy as np
import matplotlib.pyplot as plt
import random
from datautil import *

#x = np.linspace(0, 10, 1000)
x = range(2, 11)
metric = 'recall'
topk = 100
bpmf = get_avg_result('pmf', 'MovieLens2', metric, topk=topk, timeth=5)
timeSVDpp = get_avg_result('timeSVD++', 'MovieLens2', metric, topk=topk, timeth=5)
WALS = get_avg_result('weighted-als', 'MovieLens2', metric, topk=topk, timeth=5)
TensorALS = get_avg_result('tensor-als', 'MovieLens2', metric, topk=topk, timeth=5)
#BTMF = 
TRM = get_avg_result('trm', 'MovieLens2', metric, topk=topk, timeth=5)
#TTARM = [0.33, 0.47, 0.50, 0.56, 0.60, 0.65, 0.77, 0.84, 0.92]

plt.figure(figsize=(10, 8))

plt.plot(x, bpmf, "b*--", label="$BPMF$")
plt.plot(x, timeSVDpp, "d--", label="$timeSVD++$", color='seagreen')
plt.plot(x, WALS, "y^--", label="$WALS$")
plt.plot(x, TensorALS, "kh--", label="$TensorALS$")
#plt.plot(x, BTMF, "cp--", label="$BTMF$")
plt.plot(x, TRM, "rs-", label="$TRM$")
#plt.plot(x, TTARM, "m8-", label="$TTARM$")

plt.xlabel("Timestep")
#plt.ylabel("Recall@300")
plt.xlim(2, 10)
#plt.ylim(0, 1)
plt.yticks(np.arange(0, 1, 0.1))
# plt.title("models performance on Recall@300")

plt.legend(loc='upper left', numpoints=1)
plt.show()
