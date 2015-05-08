# encoding: utf-8
import sys
import os
sys.path.append(os.path.realpath(os.path.join(__file__, '../..')))
import numpy as np
from utility import fileutil

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
        r = get_avg_result(model, dataset, metric, i, timeth,
                           lambd=lambd, eta=eta)
        result.append(r[timestep-2])
    return result

def get_rating_dist(dataset):
    file_path = os.path.join(fileutil.parent_dir_of(__file__), 'data',
                             dataset, 'filtered_by_user_doc_like_list_len_10',
                             'rating_file.dat.txt')
    data = np.loadtxt(file_path)[:, 3]
    return data

if __name__ == '__main__':
    get_avg_result('pmf', 'MovieLens2', 'rmse', topk=3, timeth=5)
    get_avg_result('trm', 'MovieLens2', 'rmse', topk=3, timeth=5)
    get_avg_result('timeSVD++', 'MovieLens2', 'rmse', topk=3, timeth=5)
    get_avg_result('weighted-als', 'MovieLens2', 'rmse', topk=3, timeth=5)
    get_avg_result('tensor-als', 'MovieLens2', 'rmse', topk=3, timeth=5)
