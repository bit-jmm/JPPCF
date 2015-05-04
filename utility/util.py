import os
import math
import numpy as np
from nmf.nmf import NMF
import copy
import random
from utility import fileutil
#import scipy.io


def exec_mat_command(dir, command):
    try:
        matlab = win32com.client.GetActiveObject("Matlab.application")
    except:
        matlab = win32com.client.Dispatch("Matlab.application")
    matlab.execute('cd {}'.format(dir))
    matlab.execute(command)


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
def cal_topic_similarity_matrix(doc_word_matrix, data_path, user_num, doc_num,
                                current_user_like_dict, train=True):
    ct = np.zeros((user_num, doc_num))

    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_like_list_in_test_dict = {}
    for line in user_like_list_file.readlines():
        splits = line.strip().split()
        like_list = []
        for i in xrange(2, len(splits)):
            doc_rating = splits[i].split(':')
            like_list.append((doc_rating[0], doc_rating[1]))
        user_like_list_in_test_dict[splits[0]] = like_list
    topic_num = doc_word_matrix.shape[1]

    for user_id in current_user_like_dict:
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
        max_rating = 0.0
        for doc_rating in train_user_like_list:
            doc_id = int(doc_rating[0])
            rating = float(doc_rating[1])
            if rating > max_rating:
                max_rating = rating
            for i in range(topic_num):
                user_topic_vector[i] += rating * doc_word_matrix[doc_id, i]
        user_topic_vector = [i / (like_doc_num * max_rating) for i in user_topic_vector]

        for doc_id in range(doc_num):
            ct[user_id, doc_id] = cos_sim(user_topic_vector,
                                          list(doc_word_matrix[doc_id, :]))

    return ct


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


def generate_train_file_for_btmf(data_path, start_time, end_time):
    if os.path.exists(data_path + '/btmf_train'):
        return
    all_data_path = fileutil.parent_dir_of(fileutil.parent_dir_of(data_path))
    ratings = np.loadtxt(os.path.join(all_data_path, 'rating_file.dat.txt'))
    train_file = open(data_path + '/btmf_train', 'w')
    if start_time < end_time:
        for (user_id, doc_id, rating, timestep) in ratings:
            if int(timestep) < start_time:
                continue
            if int(timestep) >= end_time:
                break
            train_file.write('{0} {1} {2} {3}\n'.format(int(user_id)+1,
                                                        int(doc_id)+1,
                                                        rating,
                                                        int(timestep)))
    ratings = np.loadtxt(os.path.join(data_path, 'train.dat.txt'))
    for (user_id, doc_id, rating, timestep) in ratings:
        train_file.write('{0} {1} {2} {3}\n'.format(int(user_id)+1,
                                                    int(doc_id)+1,
                                                    rating,
                                                    int(timestep)))
    train_file.close()


def predict_for_btmf(model_file, user_num, doc_num, time_step):
    m_dict = scipy.io.loadmat(model_file)
    U = m_dict.get('U')
    V = m_dict.get('V')
    B = m_dict.get('B')

    predict = np.zeros((user_num, doc_num))
    for i in range(user_num):
        for j in range(doc_num):
            predict[i, j] = np.dot(V[j, :, time_step-1],
                                   (U[i, :, time_step-1]*B[:, :, i]).T)[0]
    return predict


def generate_train_and_test_file(user_num, doc_num,
                                 data_path,
                                 start_time, end_time,
                                 times, model_name):
    all_data_path = fileutil.parent_dir_of(fileutil.parent_dir_of(data_path))
    before_ratings = np.loadtxt(os.path.join(all_data_path, 'rating_file.dat.txt'))
    current_ratings = np.loadtxt(os.path.join(data_path, 'train.dat.txt'))
    i = 0
    for (user_id, doc_id, rating, timestep) in before_ratings:
        if int(timestep) >= end_time:
            break
        i += 1
    train_rating_num = i + current_ratings.shape[0]
    train_file = open(data_path + '/' + model_name + '_train' + str(times), 'w')
    train_file.write('%%MatrixMarket matrix coordinate real general\n')
    train_file.write(
        str(user_num) + ' ' + str(doc_num) + ' ' + str(train_rating_num) + '\n')
    if start_time < end_time:
        for (user_id, doc_id, rating, timestep) in before_ratings:
            if int(timestep) < start_time:
                continue
            if int(timestep) >= end_time:
                break
            train_file.write('{0} {1} {2} {3}\n'.format(int(user_id)+1,
                                                        int(doc_id)+1,
                                                        int(timestep),
                                                        int(rating)))
        for (user_id, doc_id, rating, timestep) in current_ratings:
            train_file.write('{0} {1} {2} {3}\n'.format(int(user_id)+1,
                                                        int(doc_id)+1,
                                                        int(timestep),
                                                        int(rating)))
    train_file.close()

    test_file = open(data_path + '/' + model_name + '_test' + str(times), 'w')
    test_file.write('%%MatrixMarket matrix coordinate real general\n')

    test_rating_num = user_num * doc_num
    test_file.write(
        str(user_num) + ' ' + str(doc_num) + ' ' + str(test_rating_num) + '\n')

    for i in range(user_num):
        for j in range(doc_num):
            test_file.write(str(i + 1) + ' ' + str(j + 1) + ' ' +
                    str(end_time) + ' 1\n')
    test_file.close()


def create_predict_matrix(user_num, doc_num, data_path, times, model_name):
    R = np.zeros((user_num, doc_num), dtype=float)
    predict = np.loadtxt(data_path + '/' + model_name + '_test' + str(times) + '.predict',
                         dtype=float,
                         skiprows=1)
    m, n = predict.shape
    for i in range(1, m):
        R[int(predict[i, 0]) - 1, int(predict[i, 1]) - 1] = predict[i, 2]
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
