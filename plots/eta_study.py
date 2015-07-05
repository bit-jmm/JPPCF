import numpy as np
import matplotlib.pyplot as plt
import random
from datautil import *
sys.path.append(os.path.realpath(os.path.join(__file__, '../..')))
from utility import util

#x = np.linspace(0, 10, 1000)
x = range(11)
topk = 3
dataset = 'CiteUlike2'

for metric in ['recall', 'ndcg']:
    if metric == 'ndcg':
        TTARM = [0.4905, 0.6450, 0.6975, 0.7300, 0.7280, 0.7302, 0.7415, 0.7237, 0.6871, 0.6513, 0.5840]
    else:
        TTARM = [0.3533, 0.5110, 0.5657, 0.5987, 0.6027, 0.6139, 0.6183, 0.6118, 0.5939, 0.5809, 0.5551]
    plt.figure(figsize=(10, 8))

    plt.plot(x, TTARM, "m8-", label="$TTARM$")

    plt.xlabel("Parameter eta")
    if metric == 'ndcg':
        plt.ylabel("NDCG@3")
        plt.yticks(np.arange(0.4, 0.85, 0.05))
    else:
        plt.ylabel("Recall@3")
        plt.yticks(np.arange(0.3, 0.8, 0.05))
    plt.xticks(np.arange(0, 11, 1), [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.title("models performance on Recall@300")

    plt.legend(loc='upper left', numpoints=1)
    #plt.show()
    filename = metric + '@' + str(topk) + '_eta_study.png'
    plt.savefig('/Users/jiangming/Dropbox/Research/Latex/papers/TTARM/figures/' + filename)
