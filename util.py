import numpy as np
from nmf.nmf import NMF
import ndcg
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


def cos_sim(vector_a, vector_b):
    dot_value = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for i in range(len(vector_a)):
        dot_value += vector_a[i] * vector_b[i]
        sum_a += vector_a[i] * vector_a[i]
        sum_b += vector_b[i] * vector_b[i]
    dist_a = math.sqrt(sum_a)
    dist_b = math.sqrt(sum_b)
    sim = dot_value / (dist_a * dist_b)
    return sim


# calculate topic similarity matrix
def cal_topic_similarity_matrix(W, data_path, user_num, doc_num,
                                current_user_like_dict, train=True):
    Ct = np.zeros((user_num, doc_num))

    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_like_list_in_test_dict = {}
    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(1, len(splits)):
            like_list.append(int(splits[i]))
        user_like_list_in_test_dict[int(splits[0])] = like_list
    topic_num = W.shape[1]
    for user_id in range(user_num):
        if user_id not in current_user_like_dict:
            continue
        if (not train) and (user_id not in user_like_list_in_test_dict):
            continue

        current_user_like_list = current_user_like_dict[user_id]

        if not train:
            train_user_like_list = user_like_list_in_test_dict[user_id]
        elif user_id not in user_like_list_in_test_dict:
            train_user_like_list = current_user_like_list
        else:
            user_like_list_in_test = user_like_list_in_test_dict[user_id]
            train_user_like_list = list(
                set(current_user_like_list) - set(user_like_list_in_test))
        like_doc_num = len(train_user_like_list)
        if like_doc_num == 0:
            continue
        user_topic_vector = [0] * topic_num
        for like_doc_id in train_user_like_list:
            for i in range(topic_num):
                user_topic_vector[i] += W[like_doc_id, i]
        user_topic_vector = [i / float(like_doc_num) for i in user_topic_vector]

        for doc_id in range(doc_num):
            Ct[user_id, doc_id] = cos_sim(user_topic_vector, list(W[doc_id, :]))

    return Ct


def generate_matrice_for_file(data_path, m, n):
    R = np.zeros((m, n))
    data = np.loadtxt(data_path, dtype=int)
    (row, col) = data.shape
    for i in range(row):
        R[data[i, 0] - 1, data[i, 1] - 1] = 1
    return R


def generate_matrice_for_file2(data_path, m, n):
    R = np.zeros((m, n))
    data = np.loadtxt(data_path, dtype=int)
    (row, col) = data.shape
    for i in range(row):
        R[data[i, 0], data[i, 1]] = 1
    return R


def generate_matrice_between_time(X, m, n, start_time, end_time,
                                  train_data_path=''):
    R = np.zeros((m, n))
    (row, col) = X.shape

    if start_time <= end_time:
        for i in range(row):
            if X[i, 2] < start_time:
                continue
            if X[i, 2] > end_time:
                break
            R[X[i, 0], X[i, 1]] = 1
    if train_data_path != '':
        train_data = np.loadtxt(train_data_path, dtype=int)
        (row, col) = train_data.shape
        for i in range(row):
            R[train_data[i, 0], train_data[i, 1]] = 1

    return R


def generate_matrice_between_time2(X, m, n, start_time, end_time,
                                   train_data_path):
    R = np.zeros((m, n))
    (row, col) = X.shape

    if start_time <= end_time:
        for i in range(row):
            if X[i, 2] < start_time:
                continue
            if X[i, 2] > end_time:
                break
            R[X[i, 0] - 1, X[i, 1] - 1] = 1
    if train_data_path != '':
        train_data = np.loadtxt(train_data_path, dtype=int)
        (row, col) = train_data.shape
        for i in range(row):
            R[train_data[i, 0], train_data[i, 1]] = 1

    return R


def generate_rating_list_between_time3(X, start_time, end_time,
                                       train_data_path):
    r_list = {}
    (row, col) = X.shape

    if start_time <= end_time:
        for i in range(row):
            if X[i, 2] < start_time:
                continue
            if X[i, 2] > end_time:
                break
            r_list[(X[i, 0], X[i, 1])] = (X[i, 2], 1)
    if train_data_path != '':
        train_data = np.loadtxt(train_data_path, dtype=int)
        (row, col) = train_data.shape
        for i in range(row):
            r_list[(train_data[i, 0], train_data[i, 1])] = (end_time + 1, 1)

    return r_list


def generate_train_and_test_file_for_timesvdpp(R, user_num, doc_num, data_path,
                                               user_id_dict, doc_id_dict,
                                               start_time, end_time):
    r_list = generate_rating_list_between_time3(R, start_time, end_time,
                                                data_path + '/train.dat.txt',
                                                user_id_dict, doc_id_dict)
    train_rating_num = len(r_list)
    train_file = open(data_path + '/timesvdpp_train', 'w')
    train_file.write('%%MatrixMarket matrix coordinate real general\n')
    train_file.write(
        str(user_num) + ' ' + str(doc_num) + ' ' + str(train_rating_num) + '\n')
    for i in range(user_num):
        for j in range(doc_num):
            train_file.write(
                str(i + 1) + ' ' + str(j + 1) + ' ' + str(
                    r_list.get((i + 1, j + 1), (end_time + 1, 0))[
                        0]) + ' ' + str(
                    r_list.get((i + 1, j + 1), (end_time + 1, 0))[1]) + '\n')
    train_file.close()

    test_file = open(data_path + '/timesvdpp_test', 'w')
    test_file.write('%%MatrixMarket matrix coordinate real general\n')

    test_data = np.loadtxt(data_path + '/test.dat.txt', dtype=int)
    test_rating_num = user_num * doc_num
    test_file.write(
        str(user_num) + ' ' + str(doc_num) + ' ' + str(test_rating_num) + '\n')

    for i in range(user_num):
        for j in range(doc_num):
            test_file.write(str(i + 1) + ' ' + str(j + 1) + ' ' + str(
                end_time + 1) + ' 1\n')
    test_file.close()


def create_predict_matrix(user_num, doc_num, data_path):
    R = np.zeros((user_num, doc_num), dtype=float)
    predict = np.loadtxt(data_path + '/timesvdpp_test.predict', dtype=float,
                         skiprows=1)
    m, n = predict.shape
    for i in range(1, m):
        R[predict[i, 0] - 1, predict[i, 1] - 1] = predict[i, 2]
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


def ap(rank_list):
    n = len(rank_list)
    total = 0.0
    shot_num = 0
    for i in range(n):
        if rank_list[i] == 1:
            shot_num += 1
            total += shot_num / float(i + 1)
    if shot_num == 0:
        return 0.0
    else:
        return total / shot_num


def performance_ap(Predict, data_path, at_num,
                   current_user_like_dict):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_dict = {}
    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(1, len(splits)):
            like_list.append(int(splits[i]))
        user_dict[int(splits[0])] = like_list

    (m, n) = Predict.shape

    total_ap = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        true_like_list = user_dict[user_id]
        if len(true_like_list) == 0:
            continue
        current_like_list = current_user_like_dict[user_id]
        p_like_list = list(Predict[user_id - 1, :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
            lambda x, y: cmp(y[1], x[1]))

        sort_p_like_doc_ids = []
        effective_doc_num = 0
        for i in range(n):
            if effective_doc_num == at_num:
                break
            p_doc_id = sort_p_like_list[i][0] + 1
            if (p_doc_id in current_like_list) and (
                        p_doc_id not in true_like_list):
                continue
            sort_p_like_doc_ids.append(p_doc_id)
            effective_doc_num += 1

        rank_list = []
        p_true_num = 0
        for doc_id in sort_p_like_doc_ids:
            if doc_id in true_like_list:
                rank_list.append(1)
            else:
                rank_list.append(0)
        user_ap = ap(rank_list)
        total_ap += user_ap
        effective_user_num += 1

    if effective_user_num == 0:
        avg_ap = 0
    else:
        avg_ap = total_ap / effective_user_num

    return avg_ap


def performance_ndcg(Predict, data_path, at_num,
                     current_user_like_dict):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_dict = {}
    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(1, len(splits)):
            like_list.append(int(splits[i]))
        user_dict[int(splits[0])] = like_list

    (m, n) = Predict.shape

    total_ndcg = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        true_like_list = user_dict[user_id]
        if len(true_like_list) == 0:
            continue
        current_like_list = current_user_like_dict[user_id]
        p_like_list = list(Predict[user_id - 1, :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
            lambda x, y: cmp(y[1], x[1]))

        sort_p_like_doc_ids = []
        effective_doc_num = 0
        for i in range(n):
            if effective_doc_num == at_num:
                break
            p_doc_id = sort_p_like_list[i][0] + 1
            if (p_doc_id in current_like_list) and (
                        p_doc_id not in true_like_list):
                continue
            sort_p_like_doc_ids.append(p_doc_id)
            effective_doc_num += 1

        rank_list = []
        for doc_id in sort_p_like_doc_ids:
            if doc_id in true_like_list:
                rank_list.append(1)
            else:
                rank_list.append(0)
        user_ndcg = ndcg.get_ndcg(rank_list, len(rank_list))
        total_ndcg += user_ndcg
        effective_user_num += 1

    if effective_user_num == 0:
        avg_ndcg = 0
    else:
        avg_ndcg = total_ndcg / effective_user_num

    return avg_ndcg


def performanceRMSE(Predict, Rall):
    return math.sqrt(np.mean((Predict - Rall) ** 2))


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
        p_like_list = list(Predict[user_id - 1, :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
            lambda x, y: cmp(y[1], x[1]))
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


def performance_cross_validate_recall2(Predict, data_path, recall_num,
                                       current_user_like_dict):
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
        true_like_list = user_dict[user_id]
        current_like_list = current_user_like_dict[user_id]
        p_like_list = list(Predict[user_id, :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
            lambda x, y: cmp(y[1], x[1]))

        sort_p_like_doc_ids = []
        effective_doc_num = 0
        for i in range(n):
            if effective_doc_num == recall_num:
                break
            p_doc_id = sort_p_like_list[i][0]
            if (p_doc_id in current_like_list) and (
                        p_doc_id not in true_like_list):
                continue
            sort_p_like_doc_ids.append(p_doc_id)
            effective_doc_num += 1

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
    user_like_list_file = open(
        data_path + 'user_like_list_at_time_step' + str(time_step) + '.dat.txt')
    doc_liked_list_file = open(
        data_path + 'doc_liked_list_at_time_step' + str(time_step) + '.dat.txt')

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
            p_like_list = list(Predict[user_id, :])
            p_like_dict = dict(zip(range(n), p_like_list))
            sort_p_like_list = sorted(p_like_dict.items(),
                lambda x, y: cmp(y[1], x[1]))
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
