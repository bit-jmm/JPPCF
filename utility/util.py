import math
import numpy as np
from nmf.nmf import NMF
import copy
import random


def add_list_value_for_dict(d, key, value):
    if key in d:
        d[key].append(value)
    else:
        d[key] = [value]


def reshape_matrix(matrix, row, col):
    m, n = matrix.shape
    if row == m and col == n:
        return matrix

    matrix = np.zeros((row, col))
    for i in range(m):
        for j in range(n):
            matrix[i, j] = matrix[i, j]
    return matrix


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
def cal_topic_similarity_matrix(doc_topic_matrix, data_path, user_num, doc_num,
                                current_user_like_dict):
    ctrain = np.zeros((user_num, doc_num))
    ctest = np.zeros((user_num, doc_num))

    doc_topic = doc_topic_matrix[range(doc_num), :]

    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_like_list_in_test_dict = {}
    for line in user_like_list_file.readlines():
        splits = line.strip().split()
        like_list = []
        for i in xrange(2, len(splits)):
            doc_rating = splits[i].split(':')
            like_list.append((doc_rating[0], doc_rating[1]))
        user_like_list_in_test_dict[splits[0]] = like_list
    topic_num = doc_topic_matrix.shape[1]

    rating_train = np.zeros((user_num, doc_num))
    rating_test = np.zeros((user_num, doc_num))

    for user_id in current_user_like_dict:
        if user_id in user_like_list_in_test_dict:
            like_list = user_like_list_in_test_dict[user_id]
            for doc_rating in like_list:
                doc_id = int(doc_rating[0])
                rating = float(doc_rating[1])
                rating_test[int(user_id), doc_id] = rating

    for user_id in current_user_like_dict:
        current_like_list = current_user_like_dict[user_id]
        if user_id not in user_like_list_in_test_dict:
            like_list = current_like_list
        else:
            like_list_in_test = user_like_list_in_test_dict[user_id]
            like_list = list(set(current_like_list) - set(like_list_in_test))
        for doc_rating in like_list:
            doc_id = int(doc_rating[0])
            rating = float(doc_rating[1])
            rating_train[int(user_id), doc_id] = rating

    sum_rating_train = rating_train.sum(1)
    sum_rating_test = rating_test.sum(1)

    user_topic_train = np.dot(rating_train, doc_topic)
    user_topic_test = np.dot(rating_test, doc_topic)

    for user_id in xrange(user_num):
        for topic_id in xrange(topic_num):
            if sum_rating_train[user_id] > 0:
                user_topic_train[user_id, topic_id] /= sum_rating_train[user_id]
            if sum_rating_test[user_id] > 0:
                user_topic_test[user_id, topic_id] /= sum_rating_test[user_id]

    for user_id in xrange(user_num):
        for doc_id in xrange(doc_num):
            ctrain[user_id, doc_id] = cos_sim(list(user_topic_train[user_id, :]),
                                              list(doc_topic[doc_id, :]))
            ctest[user_id, doc_id] = cos_sim(list(user_topic_test[user_id, :]),
                                              list(doc_topic[doc_id, :]))
    return (ctrain, ctest)


def generate_matrice_for_file(data_path, m, n):
    R = np.zeros((m, n))
    rating = np.loadtxt(data_path, dtype=float)
    (row, col) = rating.shape
    for i in range(row):
        user_id = int(rating[i, 0])
        doc_id = int(rating[i, 1])
        R[user_id, doc_id] = rating[i, 2]
    return R


def generate_matrice_between_time(rating, m, n, start_time, end_time,
                                  train_data_path=''):
    matrix = np.zeros((m, n))
    (row, col) = rating.shape

    if start_time <= end_time:
        for i in range(row):
            user_id = int(rating[i, 0])
            doc_id = int(rating[i, 1])
            timestep = int(rating[i, 3])
            if int(timestep) < start_time:
                continue
            if int(timestep) > end_time:
                break
            matrix[user_id, doc_id] = rating[i, 2]
    if train_data_path != '':
        train_data = np.loadtxt(train_data_path, dtype=float)
        (row, col) = train_data.shape
        for i in range(row):
            user_id = int(train_data[i, 0])
            doc_id = int(train_data[i, 1])
            matrix[user_id, doc_id] = train_data[i, 2]

    return matrix


def generate_rating_list_between_time3(rating, start_time, end_time,
                                       train_data_path):
    r_list = {}
    (row, col) = rating.shape

    if start_time <= end_time:
        for i in range(row):
            if rating[i, 2] < start_time:
                continue
            if rating[i, 2] > end_time:
                break
            r_list[(rating[i, 0], rating[i, 1])] = (rating[i, 2], 1)
    if train_data_path != '':
        train_data = np.loadtxt(train_data_path, dtype=int)
        (row, col) = train_data.shape
        for i in range(row):
            r_list[(train_data[i, 0], train_data[i, 1])] = (end_time + 1, 1)

    return r_list


def generate_train_and_test_file_for_timesvdpp(rating, user_num, doc_num,
                                               data_path,
                                               user_id_dict, doc_id_dict,
                                               start_time, end_time):
    r_list = generate_rating_list_between_time3(rating, start_time, end_time,
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


def rmse(matrix_a, matrix_b):
    return math.sqrt(np.mean((matrix_a - matrix_b) ** 2))


def norm_by_threshold(matrix, threshold):
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            if matrix[i][j] >= threshold:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    return matrix


# return random item in a item list and remove it
def random_item_from(item_list):
    n = len(item_list)
    i = random.randint(0, n-1)
    value = item_list[i]
    del item_list[i]
    return value


# split list into n folds
def random_split_list(item_list, fold_num):
    folds = {}
    n = len(item_list)
    num_of_fold = n / int(fold_num)
    for i in xrange(fold_num-1):
        for j in xrange(num_of_fold):
            item = random_item_from(item_list)
            add_list_value_for_dict(folds, i, item)
    folds[fold_num-1] = item_list
    return folds
