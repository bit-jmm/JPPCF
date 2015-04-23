import numpy as np
from utility import util


def get_user_dict(data_path):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_dict = {}
    for line in user_like_list_file.readlines():
        splits = line.strip().split()
        like_list = []
        for i in range(2, len(splits)):
            doc_rating = splits[i].split(':')
            doc_id = doc_rating[0]
            rating = doc_rating[1]
            like_list.append((doc_id, rating))
        user_dict[splits[0]] = like_list
    user_like_list_file.close()
    return user_dict


def get_doc_list(doc_rating_list):
    return [doc for (doc, rating) in doc_rating_list]


def get_rating_from_list(doc_id, doc_list, doc_rating_list):
    i = doc_list.index(doc_id)
    return float(doc_rating_list[i][1])


def get_rmse(predict_matrix, data_path):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    true_matrix = np.zeros(predict_matrix.shape, dtype=float)

    for user in user_like_list_file.readlines():
        splits = user.strip().split()
        user_id = int(splits[0])
        for i in range(2, len(splits)):
            doc_rating = splits[i].split(':')
            doc_id = int(doc_rating[0])
            rating = float(doc_rating[1])
            true_matrix[user_id][doc_id] = rating

    return util.rmse(predict_matrix, true_matrix)


def get_map(predict, data_path, at_num,
            current_user_like_dict):
    user_dict = get_user_dict(data_path)
    (m, n) = predict.shape
    total_ap = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        t_doc_rating_list = user_dict[user_id]
        t_doc_list = get_doc_list(t_doc_rating_list)
        if len(t_doc_list) == 0:
            continue
        c_doc_rating_list = current_user_like_dict[user_id]
        c_doc_list = get_doc_list(c_doc_rating_list)

        p_like_list = list(predict[int(user_id), :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
                                  lambda x, y: cmp(y[1], x[1]))
        effective_doc_num = 0
        rank_list = []
        for i in range(n):
            if effective_doc_num == at_num:
                break
            p_doc_id = str(sort_p_like_list[i][0])
            if (p_doc_id in c_doc_list) and (p_doc_id not in t_doc_list):
                continue
            if p_doc_id in t_doc_list:
                rating = get_rating_from_list(p_doc_id, t_doc_list,
                                              t_doc_rating_list)
                rank_list.append(rating)
            else:
                rank_list.append(0.0)
            effective_doc_num += 1

        user_ap = util.ap(rank_list)
        total_ap += user_ap
        effective_user_num += 1

    if effective_user_num == 0:
        avg_ap = 0
    else:
        avg_ap = total_ap / effective_user_num

    return avg_ap


def get_ndcg(predict, data_path, at_num,
             current_user_like_dict):
    user_dict = get_user_dict(data_path)
    (m, n) = predict.shape
    total_ndcg = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():

        t_doc_rating_list = user_dict[user_id]
        t_doc_list = get_doc_list(t_doc_rating_list)
        if len(t_doc_rating_list) == 0:
            continue

        c_doc_rating_list = current_user_like_dict[user_id]
        c_doc_list = get_doc_list(c_doc_rating_list)

        p_like_list = list(predict[int(user_id), :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
                                  lambda x, y: cmp(y[1], x[1]))

        effective_doc_num = 0
        rank_list = []
        for i in range(n):
            if effective_doc_num == at_num:
                break
            p_doc_id = str(sort_p_like_list[i][0])
            if (p_doc_id in c_doc_list) and (p_doc_id not in t_doc_list):
                continue
            if p_doc_id in t_doc_list:
                rating = get_rating_from_list(p_doc_id, t_doc_list,
                                              t_doc_rating_list)
                rank_list.append(rating)
            else:
                rank_list.append(0.0)
            effective_doc_num += 1
        user_ndcg = util.get_ndcg(rank_list, len(rank_list))
        total_ndcg += user_ndcg
        effective_user_num += 1

    if effective_user_num == 0:
        avg_ndcg = 0
    else:
        avg_ndcg = total_ndcg / effective_user_num

    return avg_ndcg


def get_recall(predict, data_path, recall_num, current_user_like_dict):
    user_dict = get_user_dict(data_path)
    (m, n) = predict.shape
    total_recall = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        true_like_list = get_doc_list(user_dict[user_id])
        current_like_list = get_doc_list(current_user_like_dict[user_id])
        p_like_list = list(predict[int(user_id), :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
                                  lambda x, y: cmp(y[1], x[1]))

        effective_doc_num = 0
        p_true_num = 0
        for i in range(n):
            if effective_doc_num == recall_num:
                break
            p_doc_id = str(sort_p_like_list[i][0])
            if (p_doc_id in current_like_list) and \
                    (p_doc_id not in true_like_list):
                continue
            if p_doc_id in true_like_list:
                p_true_num += 1
            effective_doc_num += 1

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