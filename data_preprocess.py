import sys
import os
import numpy as np
import csv
import datetime
import time


def generate_rating_file_by_time_interval(days):
    # set time interval (days)
    time_interval = datetime.timedelta(days)
    print 'set time_interval: ', time_interval

    data_path = './data/preprocessed_data/'
    matrices_data_path = data_path + 'data_divided_by_' + str(days) + '_days/'

    print 'start prepare data...\n\n'

    if not os.path.isdir(matrices_data_path):
        os.mkdir(matrices_data_path)

    rating_file = open(data_path + 'user_doc_rating_time.csv', 'r')
    rating_file_reader = csv.reader(rating_file)

    ratings = {}
    rating_list = []
    i = 0
    for line in rating_file_reader:
        if i == 0:
            i += 1
            continue
        rating_time = datetime.datetime.strptime(line[3].split()[0], "%Y-%m-%d")
        max_time = rating_time
        if i == 1:
            min_time = rating_time
        ratings[line[0] + ',' + line[1]] = rating_time
        rating_list.append((line[0], line[1], rating_time))
        i += 1
    rating_num = len(ratings)
    print 'rating_num: ', rating_num
    rating_file.close()

    print 'min_time: ', min_time
    print 'max_time: ', max_time

    time_step_num = (max_time - min_time).days / time_interval.days + 1
    print 'time_step_num: ', time_step_num

    time_step = 1
    rating_data_file = matrices_data_path + 'rating_data_file.dat.txt'
    writer = open(rating_data_file, 'w+')
    current_time = min_time + datetime.timedelta(days)
    i = 0
    processed_dict = {}
    for l in rating_list:
        key = l[0] + ',' + l[1]
        if key in processed_dict:
            continue
        else:
            processed_dict[key] = 1
        if i % 10000 == 0:
            print ' i= ', i
        i += 1
        if l[2] > current_time:
            time_step += 1
            current_time += datetime.timedelta(days)
        writer.write(l[0] + ' ' + l[1] + ' ' + str(time_step) + '\n')
    writer.close()

    print 'rating file process end.\n'


def get_user_like_and_doc_liked_list(time_interval):
    data_path = './data/preprocessed_data/data_divided_by_' + str(time_interval) + '_days/'
    rating_file = open(data_path + 'rating_data_file.dat.txt', 'r')

    user_writer = open(data_path + 'user_like_list.dat.txt', 'w+')
    doc_writer = open(data_path + 'doc_liked_list.dat.txt', 'w+')

    user_dict = {}
    doc_dict = {}

    for line in rating_file.readlines():
        splits = line.split()
        user_id = int(splits[0])
        doc_id = int(splits[1])

        if user_id in user_dict:
            user_dict[user_id].append(doc_id)
        else:
            user_dict[user_id] = [doc_id]
        if doc_id in doc_dict:
            doc_dict[doc_id].append(user_id)
        else:
            doc_dict[doc_id] = [user_id]
    sorted_user_list = sorted(user_dict.items(), lambda x, y: cmp(len(x[1]), len(y[1])), reverse=True)
    sorted_doc_list = sorted(doc_dict.items(), lambda x, y: cmp(len(x[1]), len(y[1])), reverse=True)

    for user in sorted_user_list:
        doc_str = ''
        for i in range(len(user[1]) - 1):
            doc_str += str(user[1][i]) + ' '
        doc_str += str(user[1][-1])
        user_writer.write(str(user[0]) + ' ' + str(len(user[1])) + ' ' + doc_str + '\n')

    for doc in sorted_doc_list:
        user_str = ''
        for i in range(len(doc[1]) - 1):
            user_str += str(doc[1][i]) + ' '
        user_str += str(doc[1][-1])
        doc_writer.write(str(doc[0]) + ' ' + str(len(doc[1])) + ' ' + user_str + '\n')

    rating_file.close()
    user_writer.close()
    doc_writer.close()


def filter_unactive_users_docs(time_interval, threshold):
    data_path = './data/preprocessed_data/data_divided_by_' +\
                str(time_interval) + '_days/'
    rating_file = open(data_path + 'rating_data_file.dat.txt', 'r')
    user_like_list = open(data_path + 'user_like_list.dat.txt', 'r')
    doc_liked_list = open(data_path + 'doc_liked_list.dat.txt', 'r')

    print 'start filter ratings whose users and docs' \
          ' like list\'s len less than ', threshold

    result_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(threshold)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    filtered_user_ids = []
    for user in user_like_list.readlines():
        splits = user.split()
        if int(splits[1]) < threshold:
            break
        filtered_user_ids.append(int(splits[0]))

    filtered_doc_ids = []
    for doc in doc_liked_list.readlines():
        splits = doc.split()
        if int(splits[1]) < threshold:
            break
        filtered_doc_ids.append(int(splits[0]))

    filter_rating_file = open(result_path + '/rating_file.dat.txt', 'w+')
    user_id_map_file = open(result_path + '/user_id_map.dat.txt', 'w+')
    doc_id_map_file = open(result_path + '/doc_id_map.dat.txt', 'w+')

    user_id_map = {}
    doc_id_map = {}
    user_dict = {}
    doc_dict = {}
    current_time_step = 1

    new_filter_user_ids = []
    new_filter_doc_ids = []

    for line in rating_file.readlines():
        splits = line.split()
        user_id = int(splits[0])
        doc_id = int(splits[1])
        time_step = int(splits[2])

        if time_step > current_time_step:
            filtered_user_like_list = open(result_path + '/user_like_list_at_time_step' +
                                           str(current_time_step) + '.dat.txt', 'w+')
            filtered_doc_liked_list = open(result_path + '/doc_liked_list_at_time_step' +
                                           str(current_time_step) + '.dat.txt', 'w+')
            sorted_user_list = sorted(user_dict.items(), lambda x, y: cmp(len(x[1]), len(y[1])), reverse=True)
            sorted_doc_list = sorted(doc_dict.items(), lambda x, y: cmp(len(x[1]), len(y[1])), reverse=True)

            for user in sorted_user_list:
                doc_str = ''
                for i in range(len(user[1]) - 1):
                    doc_str += str(user[1][i]) + ' '
                doc_str += str(user[1][-1])
                filtered_user_like_list.write(str(user[0]) + ' ' + str(len(user[1])) + ' ' + doc_str + '\n')

            for doc in sorted_doc_list:
                user_str = ''
                for i in range(len(doc[1]) - 1):
                    user_str += str(doc[1][i]) + ' '
                user_str += str(doc[1][-1])
                filtered_doc_liked_list.write(str(doc[0]) + ' ' + str(len(doc[1])) + ' ' + user_str + '\n')

            filtered_user_like_list.close()
            filtered_doc_liked_list.close()

            user_dict = {}
            doc_dict = {}
            current_time_step = time_step

        if (user_id not in filtered_user_ids) or (doc_id not in filtered_doc_ids):
            continue

        if user_id not in new_filter_user_ids:
            new_filter_user_ids.append(user_id)
        if doc_id not in new_filter_doc_ids:
            new_filter_doc_ids.append(doc_id)

        fuser_id = new_filter_user_ids.index(user_id)
        fdoc_id = new_filter_doc_ids.index(doc_id)

        if user_id not in user_id_map:
            user_id_map[user_id] = fuser_id
        if doc_id not in doc_id_map:
            doc_id_map[doc_id] = fdoc_id

        if fuser_id in user_dict:
            user_dict[fuser_id].append(fdoc_id)
        else:
            user_dict[fuser_id] = [fdoc_id]
        if fdoc_id in doc_dict:
            doc_dict[fdoc_id].append(fuser_id)
        else:
            doc_dict[fdoc_id] = [fuser_id]

        filter_rating_file.write(str(fuser_id) + ' ' + str(fdoc_id) + ' ' + str(time_step) + '\n')

    filtered_user_like_list = open(result_path + '/user_like_list_at_time_step' +
                                   str(current_time_step) + '.dat.txt', 'w+')
    filtered_doc_liked_list = open(result_path + '/doc_liked_list_at_time_step' +
                                   str(current_time_step) + '.dat.txt', 'w+')
    sorted_user_list = sorted(user_dict.items(), lambda x, y: cmp(len(x[1]), len(y[1])), reverse=True)
    sorted_doc_list = sorted(doc_dict.items(), lambda x, y: cmp(len(x[1]), len(y[1])), reverse=True)

    for user in sorted_user_list:
        doc_str = ''
        for i in range(len(user[1]) - 1):
            doc_str += str(user[1][i]) + ' '
        doc_str += str(user[1][-1])
        filtered_user_like_list.write(str(user[0]) + ' ' + str(len(user[1])) + ' ' + doc_str + '\n')

    for doc in sorted_doc_list:
        user_str = ''
        for i in range(len(doc[1]) - 1):
            user_str += str(doc[1][i]) + ' '
        user_str += str(doc[1][-1])
        filtered_doc_liked_list.write(str(doc[0]) + ' ' + str(len(doc[1])) + ' ' + user_str + '\n')

    for item in user_id_map.items():
        user_id_map_file.write(str(item[0]) + ' ' + str(item[1]) + '\n')
    for item in doc_id_map.items():
        doc_id_map_file.write(str(item[0]) + ' ' + str(item[1]) + '\n')

    user_id_map_file.close()
    doc_id_map_file.close()

    filtered_user_like_list.close()
    filtered_doc_liked_list.close()

    rating_file.close()
    user_like_list.close()
    doc_liked_list.close()
    filter_rating_file.close()

    print 'end.'


def generate_user_id_and_doc_id_map(time_interval, filter_threshold):
    data_path = './data/preprocessed_data/data_divided_by_' + str(time_interval) + '_days/'
    result_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(filter_threshold)
    rating_data = np.loadtxt(result_path + '/rating_file.dat.txt', dtype=int)

    doc_time_distribute = open(result_path + '/doc_time_distribute.dat.txt', 'w+')
    user_time_distribute = open(result_path + '/user_time_distribute.dat.txt', 'w+')

    user_id_dict = {}
    doc_id_dict = {}
    start_time = rating_data[0][2]

    row, col = rating_data.shape
    current_time = start_time
    for i in range(row):
        user_id = rating_data[i][0]
        doc_id = rating_data[i][1]
        time_step = rating_data[i][2]
        if time_step != current_time:
            user_time_distribute.write(str(current_time) + '\t' + str(len(user_id_dict)) + '\n')
            doc_time_distribute.write(str(current_time) + '\t' + str(len(doc_id_dict)) + '\n')
            current_time = time_step
        if user_id not in user_id_dict:
            user_id_dict[user_id] = 1
        if doc_id not in doc_id_dict:
            doc_id_dict[doc_id] = 1
        if i == row - 1:
            user_time_distribute.write(str(current_time) + '\t' + str(len(user_id_dict)) + '\n')
            doc_time_distribute.write(str(current_time) + '\t' + str(len(doc_id_dict)) + '\n')

    user_time_distribute.close()
    doc_time_distribute.close()


def get_doc_id_citeulike_id_map(time_interval, filter_threshold):
    origin_data_path = './data/preprocessed_data/'
    data_path = './data/preprocessed_data/data_divided_by_' + str(time_interval) + '_days/'
    result_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(filter_threshold)

    doc_id_map = open(origin_data_path + 'doc_id_citeulike_id_map.csv', 'r')
    doc_id_map_after_filter = open(result_path + '/doc_id_map.dat.txt', 'r')

    map_file = open(result_path + '/doc_id_citeulike_id_map_after_filter.dat.txt', 'w+')

    doc_id_citeulike_id_dict = {}
    i = 0
    for doc in doc_id_map.readlines():
        if i == 0:
            i += 1
            continue
        splits = doc.split(',')
        doc_id_citeulike_id_dict[int(splits[0])] = int(splits[1].strip())

    for line in doc_id_map_after_filter.readlines():
        splits = line.split()
        citeulike_id = doc_id_citeulike_id_dict[int(splits[0])]
        map_file.write(splits[1] + ' ' + str(citeulike_id) + '\n')

    doc_id_map.close()
    doc_id_map_after_filter.close()
    map_file.close()


def cross_validate_on_dict(doc_liked_list, fold_num, time_result_path):
    folds_dict_list = []
    # the last dict store the like list for docs
    # whose like list len lower than fold_num
    for i in range(fold_num + 1):
        folds_dict_list.append({})
    for doc_id in doc_liked_list.keys():
        user_ids = doc_liked_list[doc_id]
        list_len = len(user_ids)
        if list_len < fold_num:
            folds_dict_list[fold_num][doc_id] = user_ids
        else:
            step_len = list_len / fold_num
            remainder = list_len % fold_num
            for i in range(fold_num):
                for j in range(i * step_len, (i + 1) * step_len):
                    if folds_dict_list[i].has_key(doc_id):
                        folds_dict_list[i][doc_id].append(user_ids[j])
                    else:
                        folds_dict_list[i][doc_id] = [user_ids[j]]
                for j in range(list_len - remainder, list_len):
                    if folds_dict_list[i].has_key(doc_id):
                        folds_dict_list[i][doc_id].append(user_ids[j])
                    else:
                        folds_dict_list[i][doc_id] = [user_ids[j]]
    # end evenly split user-doc pair to folds
    # start generate train_data and test_data
    for i in range(fold_num):
        data_path = time_result_path + '/data_' + str(i)
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
        train_file = open(data_path + '/train.dat.txt', 'w+')
        test_file = open(data_path + '/test.dat.txt', 'w+')
        user_like_list_file = open(data_path + '/user_like_list_in_test.dat.txt', 'w+')

        user_like_list_dict_in_test = {}

        doc_dict = folds_dict_list[i]
        for doc_id in doc_dict.keys():
            for user_id in doc_dict[doc_id]:
                if user_like_list_dict_in_test.has_key(user_id):
                    user_like_list_dict_in_test[user_id].append(doc_id)
                else:
                    user_like_list_dict_in_test[user_id] = [doc_id]
                test_file.write(str(user_id) + ' ' + str(doc_id) + '\n')
        for fold_id in range(fold_num + 1):
            if fold_id == i:
                continue
            doc_dict = folds_dict_list[fold_id]
            for doc_id in doc_dict.keys():
                for user_id in doc_dict[doc_id]:
                    train_file.write(str(user_id) + ' ' + str(doc_id) + '\n')

        for user_id in user_like_list_dict_in_test:
            temp = str(user_id) + ' ' + ' '.join(str(d) for d in user_like_list_dict_in_test[user_id]) + '\n'
            user_like_list_file.write(temp)

        train_file.close()
        test_file.close()
        user_like_list_file.close()


def generate_cross_validate_data(time_interval, filter_threshold, fold_num=5):
    data_path = './data/preprocessed_data/data_divided_by_' + str(time_interval) + '_days/'
    result_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(filter_threshold)
    rating_data = np.loadtxt(result_path + '/rating_file.dat.txt', dtype=int)
    start_time = rating_data[0][2]
    end_time = rating_data[-1][2]

    # generate data at every time step
    for current_time_step in range(start_time, end_time + 1):
        time_path = result_path + '/time_step_' + str(current_time_step)
        if not os.path.isdir(time_path):
            os.mkdir(time_path)
        doc_liked_file = open(result_path + '/doc_liked_list_at_time_step' + str(current_time_step) + '.dat.txt', 'r')
        doc_liked_dict = {}
        for line in doc_liked_file.readlines():
            splits = line.split()
            doc_id = int(splits[0])
            users = [int(splits[uid]) for uid in range(2, len(splits))]
            doc_liked_dict[doc_id] = users

        cross_validate_on_dict(doc_liked_dict, fold_num, time_path)


if __name__ == '__main__':
    start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print 'process start at : ', start
    print '\n'
    time_interval = 360
    filter_threshold = 10
    fold_num = 5

    generate_rating_file_by_time_interval(time_interval)
    get_user_like_and_doc_liked_list(time_interval)
    filter_unactive_users_docs(time_interval, filter_threshold)
    generate_user_id_and_doc_id_map(time_interval, filter_threshold)
    get_doc_id_citeulike_id_map(time_interval, filter_threshold)
    generate_cross_validate_data(time_interval, filter_threshold, fold_num)

    import time

    end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print 'process end at : ', end

    print 'process take time : ', str(
        datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S'))