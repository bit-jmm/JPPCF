import numpy as np
# import scipy.sparse as sp
from nmf.nmf import NMF
import math


def reshape_matrix(matrix, row, col):
    m, n = matrix.shape
    if row == m and col == n:
        return matrix

    R = np.zeros((row, col))
    for i in range(m):
        for j in range(n):
            R[i, j] = matrix[i, j]
    return R

def generate_matrice_for_file(data_path, m, n):
    R = np.zeros((m,n))
    data = np.loadtxt(data_path, dtype=int)
    (row, col) = data.shape
    for i in range(row):
        R[data[i, 0]-1, data[i, 1]-1] = 1
    return R

def generate_matrice_for_file2(data_path, m, n, user_id_dict, doc_id_dict):
    R = np.zeros((m,n))
    data = np.loadtxt(data_path, dtype=int)
    (row, col) = data.shape
    for i in range(row):
        R[user_id_dict[data[i, 0]]-1, doc_id_dict[data[i, 1]]-1] = 1
    return R

def generate_matrice_between_time(X, m, n, start_time, end_time, train_data_path = ''):
    R = np.zeros((m, n))
    (row, col) = X.shape

    if start_time <= end_time:
        for i in range(row):
            if X[i, 2] < start_time:
                continue
            if X[i, 2] > end_time:
                break
            R[X[i, 0]-1, X[i, 1]-1] = 1
    if train_data_path != '':
        train_data = np.loadtxt(train_data_path, dtype=int)
        (row, col) = train_data.shape
        for i in range(row):
            R[train_data[i, 0]-1, train_data[i, 1]-1] = 1

    return R

def generate_matrice_between_time2(X, m, n, start_time, end_time,
                                   train_data_path, user_id_dict,
                                   doc_id_dict):
    R = np.zeros((m, n))
    (row, col) = X.shape

    if start_time <= end_time:
        for i in range(row):
            if X[i, 2] < start_time:
                continue
            if X[i, 2] > end_time:
                break
            R[X[i, 0]-1, X[i, 1]-1] = 1
    if train_data_path != '':
        train_data = np.loadtxt(train_data_path, dtype=int)
        (row, col) = train_data.shape
        for i in range(row):
            R[user_id_dict[train_data[i, 0]]-1, doc_id_dict[train_data[i, 1]]-1] = 1

    return R

def nmf(A, k=10, iter_num=100, epsilon=0.01, calc_error=True,
        calc_error_num=10):
    nmf = NMF()
    nmf.setup(A, k, iter_num, epsilon, calc_error, calc_error_num)
    nmf.run()
    return (nmf.W, nmf.H)

def avg_of_list(list_data):
    total = 0.0
    avg = 0.0
    for i in list_data:
        total += i
    if len(list_data) > 0:
        avg = total / len(list_data)
    return avg

def performanceRMSE(Predict, Rall):
    return math.sqrt(np.mean((Predict - Rall)**2))

def performance_cross_validate_recall(Predict, data_path, recall_num):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_dict = {}
    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(1, len(splits)):
            like_list.append(int(splits[i]))
        user_dict[int(splits[0])] = like_list

    (m, n) = Predict.shape
    
    total_recall = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        p_like_list = list(Predict[user_id-1,:])
        p_like_dict = dict(zip(range(n),p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(), lambda x,y: cmp(y[1],x[1]))
        sort_p_like_doc_ids = []
        for i in range(recall_num):
            sort_p_like_doc_ids.append(sort_p_like_list[i][0] + 1)

        true_like_list = user_dict[user_id] 
        p_true_num = 0
        for doc_id in true_like_list:
            if doc_id in sort_p_like_doc_ids:
                p_true_num += 1

        user_recall = 0
        if len(true_like_list) > 0:
            user_recall = float(p_true_num) / len(true_like_list)
        total_recall += user_recall
        effective_user_num += 1 

    if effective_user_num == 0:
        avg_recall = 0
    else:
        avg_recall = total_recall / effective_user_num

    return avg_recall


def performance_cross_validate_recall2(Predict, data_path, recall_num, user_id_dict, doc_id_dict):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_dict = {}
    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(1, len(splits)):
            like_list.append(doc_id_dict[int(splits[i])])
        user_dict[user_id_dict[int(splits[0])]] = like_list

    (m, n) = Predict.shape
    
    total_recall = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        p_like_list = list(Predict[user_id-1,:])
        p_like_dict = dict(zip(range(n),p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(), lambda x,y: cmp(y[1],x[1]))
        sort_p_like_doc_ids = []
        for i in range(recall_num):
            sort_p_like_doc_ids.append(sort_p_like_list[i][0] + 1)

        true_like_list = user_dict[user_id] 
        p_true_num = 0
        for doc_id in true_like_list:
            if doc_id in sort_p_like_doc_ids:
                p_true_num += 1

        user_recall = 0
        if len(true_like_list) > 0:
            user_recall = float(p_true_num) / len(true_like_list)
        total_recall += user_recall
        effective_user_num += 1 

    if effective_user_num == 0:
        avg_recall = 0
    else:
        avg_recall = total_recall / effective_user_num

    return avg_recall

def performance_recall(Predict, data_path, time_step, recall_num):
    user_like_list_file = open(data_path + 'user_like_list_at_time_step' + str(time_step) + '.dat.txt') 
    doc_liked_list_file = open(data_path + 'doc_liked_list_at_time_step' + str(time_step) + '.dat.txt') 

    user_dict = {}

    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(2, len(splits)):
            like_list.append(int(splits[i]))
        user_dict[int(splits[0])] = like_list

    (m, n) = Predict.shape
    
    total_recall = 0.0
    effective_user_num = 0
    for user_id in range(m):
        if user_dict.has_key(user_id + 1):
            p_like_list = list(Predict[user_id,:])
            p_like_dict = dict(zip(range(n),p_like_list))
            sort_p_like_list = sorted(p_like_dict.items(), lambda x,y: cmp(y[1],x[1]))
            sort_p_like_doc_ids = []
            for i in range(recall_num):
                sort_p_like_doc_ids.append(sort_p_like_list[i][0] + 1)

            true_like_list = user_dict[user_id + 1] 
            p_true_num = 0
            for doc_id in true_like_list:
                if doc_id in sort_p_like_doc_ids:
                    p_true_num += 1

            user_recall = 0
            if len(true_like_list) > 0:
                user_recall = float(p_true_num) / len(true_like_list)
            total_recall += user_recall
            effective_user_num += 1 

    if effective_user_num == 0:
        avg_recall = 0
    else:
        avg_recall = total_recall / effective_user_num

    return avg_recall

def norm_by_threshold(matrix, threshold):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            if matrix[i][j] >= threshold:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    return matrix
