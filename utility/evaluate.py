import numpy as np
from utility import util


def get_rmse(predict_matrix, data_path):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')

    true_matrix = np.zeros(predict_matrix.shape)

    for user in user_like_list_file.readlines():
        splits = user.split()
        user_id = int(splits[0])
        for i in range(1, len(splits)):
            true_matrix[user_id][int(splits[i])] = 1

    return util.rmse(predict_matrix, true_matrix)


def get_map(predict, data_path, at_num,
                    current_user_like_dict):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_dict = {}
    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(1, len(splits)):
            like_list.append(int(splits[i]))
        user_dict[int(splits[0])] = like_list

    (m, n) = predict.shape

    total_ap = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        true_like_list = user_dict[user_id]
        if len(true_like_list) == 0:
            continue
        current_like_list = current_user_like_dict[user_id]
        p_like_list = list(predict[user_id, :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
            lambda x, y: cmp(y[1], x[1]))

        sort_p_like_doc_ids = []
        effective_doc_num = 0
        for i in range(n):
            if effective_doc_num == at_num:
                break
            p_doc_id = sort_p_like_list[i][0]
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
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_dict = {}
    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(1, len(splits)):
            like_list.append(int(splits[i]))
        user_dict[int(splits[0])] = like_list

    (m, n) = predict.shape

    total_ndcg = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        true_like_list = user_dict[user_id]
        if len(true_like_list) == 0:
            continue
        current_like_list = current_user_like_dict[user_id]
        p_like_list = list(predict[user_id, :])
        p_like_dict = dict(zip(range(n), p_like_list))
        sort_p_like_list = sorted(p_like_dict.items(),
            lambda x, y: cmp(y[1], x[1]))

        sort_p_like_doc_ids = []
        effective_doc_num = 0
        for i in range(n):
            if effective_doc_num == at_num:
                break
            p_doc_id = sort_p_like_list[i][0]
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
        user_ndcg = util.get_ndcg(rank_list, len(rank_list))
        total_ndcg += user_ndcg
        effective_user_num += 1

    if effective_user_num == 0:
        avg_ndcg = 0
    else:
        avg_ndcg = total_ndcg / effective_user_num

    return avg_ndcg


def get_recall(predict, data_path, recall_num,
                       current_user_like_dict):
    user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt')
    user_dict = {}
    for user in user_like_list_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(1, len(splits)):
            like_list.append(int(splits[i]))
        user_dict[int(splits[0])] = like_list

    (m, n) = predict.shape

    total_recall = 0.0
    effective_user_num = 0
    for user_id in user_dict.keys():
        true_like_list = user_dict[user_id]
        current_like_list = current_user_like_dict[user_id]
        p_like_list = list(predict[user_id, :])
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