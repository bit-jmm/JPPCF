import numpy as np
import matplotlib.pyplot as plt
import random
from datautil import *
sys.path.append(os.path.realpath(os.path.join(__file__, '../..')))
from utility import util

#x = np.linspace(0, 10, 1000)
x = range(8)
topk = 3
dataset = 'CiteUlike2'

for metric in ['recall', 'ndcg']:
    if metric == 'ndcg':
        TRM = [0.4907, 0.4963, 0.4950, 0.4985, 0.4965, 0.5022, 0.4993, 0.4903]
    else:
        TRM = [0.3567, 0.3592, 0.3582, 0.3612, 0.3625, 0.3646, 0.3626, 0.3550]
    plt.figure(figsize=(10, 8))

    plt.plot(x, TRM, "rs-", label="$TRM$")

    plt.xlabel("Parameter lambda")
    if metric == 'recall':
        plt.ylabel("Recall@3")
        plt.yticks(np.arange(0.3425, 0.3750, 0.005))
    else:
        plt.ylabel("NDCG@3")
        plt.yticks(np.arange(0.48, 0.515, 0.005))
    plt.xticks(np.arange(0, 8, 1), [0.0001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
    # plt.title("models performance on Recall@300")

    plt.legend(loc='upper left', numpoints=1)
    #plt.show()
    filename = metric + '@' + str(topk) + '_lambda_study.png'
    plt.savefig('/Users/jiangming/Dropbox/Research/Latex/papers/TTARM/figures/' + filename)
