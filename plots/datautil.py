# encoding: utf-8
import sys
import os
sys.path.append(os.path.realpath(os.path.join(__file__, '../..')))
import numpy as np
from utility import fileutil
from utility import util

def get_avg_result(model, dataset, metric, topk=3, fold=5, time_step=-1,
                   filter_num=10, eta=0.3, k=20, lambd=10, alpha=0.05, timeth=1):
    total_result = [0] * 9
    result_path = os.path.join(fileutil.parent_dir_of(__file__), 'result')
    for i in range(1, timeth+1):
        if model=='ttarm':
            result_file = os.path.join(result_path,
                            'new_{0}_time_step_{1}_filter_by_{2}'.format(model,
                                360, filter_num),
                            'eta_{0}_fold_{1}_k_{2}_lambda_{3}_alpha_{4}'.format(eta,
                                fold, k, lambd, alpha),
                            metric, '{0}_at_{1}.txt'.format(metric, topk))
        elif model == 'trm':
            result_file = os.path.join(result_path,
                            'new_{0}_time_step_{1}_filter_by_{2}'.format(model,
                                time_step, filter_num),
                            'fold_{0}_k_{1}_lambda_{2}_{3}_{4}'.format(fold, k,
                                lambd, dataset, i),
                            metric, '{0}_at_{1}.txt'.format(metric, topk))
        else:
            result_file = os.path.join(result_path,
                            '{0}_time_step_{1}_filter_by_{2}'.format(model,
                                time_step, filter_num),
                            'fold_{0}_k_{1}_{2}_{3}'.format(fold, k,
                                dataset, i),
                            metric, '{0}_at_{1}.txt'.format(metric, topk))
        result = np.loadtxt(result_file, skiprows=1)
        result = list(result)
        for j in range(9):
            total_result[j] = total_result[j] + result[j]
    avg_result = [value/timeth for value in total_result]
    return avg_result

def get_result_at_time(model, dataset, metric, timestep, timeth,
                       lambd=10, eta=0.3):
    result = []
    for i in [3, 10, 50, 100, 300, 500, 1000]:
        r = get_avg_result(model, dataset, metric, i, timeth=timeth,
                           lambd=lambd, eta=eta)
        result.append(r[timestep-2])
    return result

def get_rating_dist(dataset):
    file_path = os.path.join(fileutil.parent_dir_of(__file__), 'data',
                             dataset, 'filtered_by_user_doc_like_list_len_10',
                             'rating_file.dat.txt')
    data = np.loadtxt(file_path)[:, 3]
    return data

def get_mean_std_result(model, dataset, metric, topk=3, fold=5, time_step=-1,
                   filter_num=10, eta=0.3, k=20, lambd=10, alpha=0.05, timeth=1):
    result_path = os.path.join(fileutil.parent_dir_of(__file__), 'result')
    total_result = []
    for i in range(1, timeth+1):
        if model=='ttarm':
            result_file = os.path.join(result_path,
                            'new_{0}_time_step_{1}_filter_by_{2}'.format(model,
                                360, filter_num),
                            'eta_{0}_fold_{1}_k_{2}_lambda_{3}_alpha_{4}'.format(eta,
                                fold, k, lambd, alpha),
                            metric, '{0}_at_{1}.txt'.format(metric, topk))
        elif model == 'trm':
            result_file = os.path.join(result_path,
                            'new_{0}_time_step_{1}_filter_by_{2}'.format(model,
                                time_step, filter_num),
                            'fold_{0}_k_{1}_lambda_{2}_{3}_{4}'.format(fold, k,
                                lambd, dataset, i),
                            metric, '{0}_at_{1}.txt'.format(metric, topk))
        else:
            result_file = os.path.join(result_path,
                            '{0}_time_step_{1}_filter_by_{2}'.format(model,
                                time_step, filter_num),
                            'fold_{0}_k_{1}_{2}_{3}'.format(fold, k,
                                dataset, i),
                            metric, '{0}_at_{1}.txt'.format(metric, topk))
        result = np.loadtxt(result_file, skiprows=1)
        result = list(result)
        total_result.append(result)
    total_result = np.array(total_result)
    mean = list(np.mean(total_result, 0))
    std = list(np.std(total_result, 0))

    return (mean, std)


if __name__ == '__main__':
    for model in ['pmf', 'weighted-als', 'timeSVD++', 'tensor-als', 'BTMF', 'trm', 'ttarm']:
        if model == 'ttarm':
            datasets = ['CiteUlike2']
        else:
            datasets = ['CiteUlike2', 'MovieLens2']
        if model == 'BTMF':
            timeth = 3
        else:
            timeth = 5
        for dataset in datasets:
            for metric in ['ndcg']:
                result = get_mean_std_result(model, dataset, metric, topk=100, timeth=timeth)
                print '\n\n{0} on {1} for {2}\n\t'.format(model, dataset, metric)
                print '\nmean: '
                for i in range(5):
                    print '\t%.4f' % result[0][i],
                print '\nstd error: '
                for i in range(5):
                    print '\t%.4f' % result[1][i],

