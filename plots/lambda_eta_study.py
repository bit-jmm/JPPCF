import numpy as np
import matplotlib.pyplot as plt
import random
from datautil import *
sys.path.append(os.path.realpath(os.path.join(__file__, '../..')))
from utility import util

#x = np.linspace(0, 10, 1000)
x = range(5)
topk = 100
dataset = 'CiteUlike2'

for metric in ['recall', 'ndcg']:
    TTARM = []
    for lambd in [0.001, 0.1, 10, 100, 10000]:
        result = get_avg_result('ttarm', dataset, metric, topk=topk,
                                timeth=5, lambd=lambd)
        TTARM.append(util.avg_of_list(result))

    plt.figure(figsize=(10, 8))

    plt.plot(x, TTARM, "m8-", label="$TTARM$")

    #plt.xlabel("Timestep")
    #plt.ylabel("Recall@300")
    if metric == 'recall':
        plt.yticks(np.arange(0.725, 0.8, 0.0125))
    else:
        plt.yticks(np.arange(0.675, 0.775, 0.0125))
    plt.xticks(np.arange(0, 5, 1), [0.001, 0.1, 10, 100, 10000])
    # plt.title("models performance on Recall@300")

    plt.legend(loc='upper left', numpoints=1)
    #plt.show()
    filename = metric + '@' + str(topk) + '_lambda_study.png'
    plt.savefig('/Users/jiangming/Dropbox/Research/Latex/papers/TTARM/figures/' + filename)

for metric in ['recall', 'ndcg']:
    TTARM = []
    for eta in [0.1, 0.3, 0.5, 0.7, 0.9]:
        result = get_avg_result('ttarm', dataset, metric, topk=topk,
                                timeth=5, eta=eta)
        TTARM.append(util.avg_of_list(result))

    plt.figure(figsize=(10, 8))

    plt.plot(x, TTARM, "m8-", label="$TTARM$")

    #plt.xlabel("Timestep")
    #plt.ylabel("Recall@300")
    if metric == 'recall':
        plt.yticks(np.arange(0.7, 0.8, 0.0125))
    else:
        plt.yticks(np.arange(0.6, 0.8, 0.025))

    plt.xticks(np.arange(0, 5, 1), [0.01, 0.3, 0.5, 0.7, 0.9])
    # plt.title("models performance on Recall@300")

    plt.legend(loc='upper left', numpoints=1)
    #plt.show()
    filename = metric + '@' + str(topk) + '_eta_study.png'
    plt.savefig('/Users/jiangming/Dropbox/Research/Latex/papers/TTARM/figures/' + filename)
