import os
import numpy as np
import csv
import datetime
import time
from utility import util
from utility import fileutil

class PrepareData:

    def __init__(self, data_path, filter_threshold,
                 fold_num=5, time_interval_days=-1):
        self.__data_path = data_path + '/'
        self.__filter_threshold = filter_threshold
        self.__time_interval_days = time_interval_days
        self.__fold_num = fold_num

    def generate_rating_file_by_time_interval(self):
        if self.__time_interval_days < 0:
            return
        time_interval = datetime.timedelta(self.__time_interval_days)
        print 'set time_interval: ', time_interval

        matrices_data_path = self.__data_path + 'data_divided_by_' + \
                                                str(self.__time_interval_days) +\
                                                '_days/'

        print 'start prepare data...\n\n'

        if not os.path.isdir(matrices_data_path):
            os.mkdir(matrices_data_path)

        rating_file = open(self.__data_path + 'user_doc_rating_time.csv', 'r')
        rating_file_reader = csv.reader(rating_file)

        ratings = {}
        rating_list = []
        i = 0
        min_time = 0
        max_time = 0
        for line in rating_file_reader:
            if i == 0:
                i += 1
                continue
            rating_time = datetime.datetime.strptime(line[3].split()[0],
                                                     "%Y-%m-%d")
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
        current_time = min_time + time_interval
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
                current_time += time_interval
            writer.write(l[0] + ' ' + l[1] + ' 1 ' + str(time_step) + '\n')
        writer.close()

        print 'rating file process end.\n'

    def get_user_like_and_doc_liked_list(self):
        if self.__time_interval_days > 0:
            data_path = self.__data_path + 'data_divided_by_' + \
                                           str(self.__time_interval_days) +\
                                           '_days/'
        else:
            data_path = self.__data_path

        rating_file = open(data_path + 'rating_data_file.dat.txt', 'r')
        user_writer = open(data_path + 'user_like_list.dat.txt', 'w+')
        doc_writer = open(data_path + 'doc_liked_list.dat.txt', 'w+')

        user_dict = {}
        doc_dict = {}

        for line in rating_file.readlines():
            splits = line.split()
            user_id = splits[0]
            doc_id = splits[1]
            rating = splits[2]
            doc_rating = str.format('{0}:{1}', doc_id, rating)
            user_rating = str.format('{0}:{1}', user_id, rating)

            if user_id in user_dict:
                user_dict[user_id].append(doc_rating)
            else:
                user_dict[user_id] = [doc_rating]
            if doc_id in doc_dict:
                doc_dict[doc_id].append(user_rating)
            else:
                doc_dict[doc_id] = [user_rating]

        sorted_user_list = sorted(user_dict.items(),
                                  lambda x, y: cmp(len(x[1]), len(y[1])),
                                  reverse=True)
        sorted_doc_list = sorted(doc_dict.items(),
                                 lambda x, y: cmp(len(x[1]), len(y[1])),
                                 reverse=True)

        for user in sorted_user_list:
            doc_rating_chain = ' '.join([doc_rating for doc_rating in user[1]])
            user_writer.write(str.format('{0} {1} {2}\n',
                                         user[0], len(user[1]),
                                         doc_rating_chain))

        for doc in sorted_doc_list:
            user_rating_chain = \
                ' '.join([user_rating for user_rating in doc[1]])
            doc_writer.write(str.format('{0} {1} {2}\n',
                                        doc[0], len(doc[1]), user_rating_chain))

        rating_file.close()
        user_writer.close()
        doc_writer.close()

    def write_user_doc_like_file_at_time(self, result_path, user_dict,
                                 doc_dict, current_time_step):
        filtered_user_like_list = open(result_path +
                                       '/user_like_list_at_time_step' +
                                       str(current_time_step) +
                                       '.dat.txt', 'w+')
        filtered_doc_liked_list = open(result_path +
                                       '/doc_liked_list_at_time_step' +
                                       str(current_time_step) +
                                       '.dat.txt', 'w+')

        sorted_user_list = sorted(user_dict.items(),
                                  lambda x, y: cmp(len(x[1]), len(y[1])),
                                  reverse=True)
        sorted_doc_list = sorted(doc_dict.items(),
                                 lambda x, y: cmp(len(x[1]), len(y[1])),
                                 reverse=True)

        for user in sorted_user_list:
            doc_rating_chain =\
                ' '.join([str(doc_rating) for doc_rating in user[1]])
            filtered_user_like_list.write(str.format('{0} {1} {2}\n',
                                                     user[0],
                                                     len(user[1]),
                                                     doc_rating_chain))
        for doc in sorted_doc_list:
            user_rating_chain = \
                ' '.join([str(user_rating) for user_rating in doc[1]])
            filtered_doc_liked_list.write(str.format('{0} {1} {2}\n',
                                                     doc[0],
                                                     len(doc[1]),
                                                     user_rating_chain))

        filtered_user_like_list.close()
        filtered_doc_liked_list.close()

    def filter_unactive_users_docs(self):
        threshold = self.__filter_threshold

        if self.__time_interval_days > 0:
            data_path = self.__data_path + 'data_divided_by_' + \
                                           str(self.__time_interval_days) +\
                                           '_days/'
        else:
            data_path = self.__data_path

        rating_file = open(data_path + 'rating_data_file.dat.txt', 'r')
        user_like_list = open(data_path + 'user_like_list.dat.txt', 'r')
        doc_liked_list = open(data_path + 'doc_liked_list.dat.txt', 'r')

        print 'start filter ratings whose users and docs' \
              ' like list\'s len less than ', threshold

        result_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(
            threshold)

        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        filtered_user_ids = []
        for user in user_like_list.readlines():
            splits = user.split()
            if int(splits[1]) < threshold:
                break
            filtered_user_ids.append(splits[0])

        filtered_doc_ids = []
        for doc in doc_liked_list.readlines():
            splits = doc.split()
            if int(splits[1]) < threshold:
                break
            filtered_doc_ids.append(splits[0])

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
            user_id = splits[0]
            doc_id = splits[1]
            rating = splits[2]
            time_step = int(splits[3])

            if time_step > current_time_step:
                self.write_user_doc_like_file_at_time(result_path, user_dict,
                                              doc_dict, current_time_step)
                user_dict = {}
                doc_dict = {}
                current_time_step = time_step

            if (user_id not in filtered_user_ids) or\
                    (doc_id not in filtered_doc_ids):
                continue

            if user_id not in new_filter_user_ids:
                new_filter_user_ids.append(user_id)
            if doc_id not in new_filter_doc_ids:
                new_filter_doc_ids.append(doc_id)

            fuser_id = str(new_filter_user_ids.index(user_id))
            fdoc_id = str(new_filter_doc_ids.index(doc_id))

            if user_id not in user_id_map:
                user_id_map[user_id] = fuser_id
            if doc_id not in doc_id_map:
                doc_id_map[doc_id] = fdoc_id

            doc_rating = str.format('{0}:{1}', fdoc_id, rating)
            user_rating = str.format('{0}:{1}', fuser_id, rating)

            if fuser_id in user_dict:
                user_dict[fuser_id].append(doc_rating)
            else:
                user_dict[fuser_id] = [doc_rating]
            if fdoc_id in doc_dict:
                doc_dict[fdoc_id].append(user_rating)
            else:
                doc_dict[fdoc_id] = [user_rating]

            filter_rating_file.write(str.format('{0} {1} {2} {3}\n',
                                                fuser_id, fdoc_id,
                                                rating, time_step))

        self.write_user_doc_like_file_at_time(result_path, user_dict,
                                      doc_dict, current_time_step)

        for item in user_id_map.items():
            user_id_map_file.write(str(item[0]) + ' ' + str(item[1]) + '\n')
        for item in doc_id_map.items():
            doc_id_map_file.write(str(item[0]) + ' ' + str(item[1]) + '\n')

        user_id_map_file.close()
        doc_id_map_file.close()
        rating_file.close()
        user_like_list.close()
        doc_liked_list.close()
        filter_rating_file.close()

        print 'end.'

    def generate_users_and_docs_dist(self):
        if self.__time_interval_days > 0:
            data_path = self.__data_path + 'data_divided_by_' +\
                                           str(self.__time_interval_days) +\
                                           '_days/'
        else:
            data_path = self.__data_path

        result_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(
            self.__filter_threshold)
        rating_data = np.loadtxt(result_path + '/rating_file.dat.txt',
                                 dtype=float)

        doc_time_distribute = open(result_path + '/doc_time_distribute.dat.txt',
                                   'w+')
        user_time_distribute = open(
            result_path + '/user_time_distribute.dat.txt', 'w+')

        user_id_dict = {}
        doc_id_dict = {}
        start_time = int(rating_data[0][3])
        row, col = rating_data.shape
        current_time = start_time

        for i in xrange(row):
            user_id = int(rating_data[i][0])
            doc_id = int(rating_data[i][1])
            time_step = int(rating_data[i][3])
            if time_step != current_time:
                user_time_distribute.write(
                    str(current_time) + '\t' + str(len(user_id_dict)) + '\n')
                doc_time_distribute.write(
                    str(current_time) + '\t' + str(len(doc_id_dict)) + '\n')
                current_time = time_step
            if user_id not in user_id_dict:
                user_id_dict[user_id] = 1
            if doc_id not in doc_id_dict:
                doc_id_dict[doc_id] = 1
            if i == row - 1:
                user_time_distribute.write(
                    str(current_time) + '\t' + str(len(user_id_dict)) + '\n')
                doc_time_distribute.write(
                    str(current_time) + '\t' + str(len(doc_id_dict)) + '\n')

        user_time_distribute.close()
        doc_time_distribute.close()

    def get_doc_id_citeulike_id_map(self):
        if self.__time_interval_days < 0:
            return
        origin_data_path = self.__data_path
        data_path = origin_data_path + 'data_divided_by_' + \
                                       str(self.__time_interval_days) + '_days/'
        result_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(
            self.__filter_threshold)

        doc_id_map = open(origin_data_path + 'doc_id_citeulike_id_map.csv', 'r')
        doc_id_map_after_filter = open(result_path + '/doc_id_map.dat.txt', 'r')

        map_file = open(
            result_path + '/doc_id_citeulike_id_map_after_filter.dat.txt', 'w+')

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

    def cross_validate_on_ratings(self, ratings, time_result_path,
                                  current_time_step):
        fold_num = self.__fold_num
        fold_dict = util.random_split_list(ratings, fold_num)

        # end evenly split user-doc pair to folds
        # start generate train_data and test_data
        for i in xrange(fold_num):
            data_path = time_result_path + '/data_' + str(i)
            if not os.path.isdir(data_path):
                os.mkdir(data_path)
            train_file = open(data_path + '/train.dat.txt', 'w+')
            test_file = open(data_path + '/test.dat.txt', 'w+')
            cold_test_file = open(data_path + '/cold_test.dat.txt', 'w+')

            user_like_list_file = open(
                data_path + '/user_like_list_in_test.dat.txt', 'w+')
            user_like_list_cold_file = open(
                data_path + '/user_like_list_in_cold_test.dat.txt', 'w+')

            # generate train file
            doc_ids_in_train = []
            for fold_id in xrange(fold_num):
                if fold_id == i:
                    continue
                ratings_in_fold = fold_dict[fold_id]
                for (user_id, doc_id, rating) in ratings_in_fold:
                    doc_ids_in_train.append(doc_id)
                    train_file.write(str.format('{0} {1} {2} {3}\n',
                                                user_id, doc_id,
                                                rating, current_time_step))
            # generate whole test file and
            # test file for evaluate cold start problem
            doc_dist_path = \
                os.path.join(fileutil.parent_dir_of(time_result_path),
                             'doc_time_distribute.dat.txt')
            max_doc_id = fileutil.num_in_file(doc_dist_path,
                                              current_time_step-1, 2)
            user_like_list_dict_in_test = {}
            user_like_list_dict_in_cold_test = {}

            ratings_in_fold = fold_dict[i]
            for (user_id, doc_id, rating) in ratings_in_fold:
                doc_rating = doc_id + ':' + rating
                util.add_list_value_for_dict(user_like_list_dict_in_test,
                                             user_id, doc_rating)
                if int(doc_id) >= max_doc_id and\
                   doc_id not in doc_ids_in_train:
                    cold_test_file.write(str.format('{0} {1} {2} {3}\n',
                                                    user_id, doc_id, rating,
                                                    current_time_step))
                    util.add_list_value_for_dict(
                        user_like_list_dict_in_cold_test,
                        user_id, doc_rating)
                test_file.write(str.format('{0} {1} {2} {3}\n',
                                           user_id, doc_id, rating,
                                           current_time_step))

            for user_id in user_like_list_dict_in_test.keys():
                temp = user_id + ' ' +\
                       str(len(user_like_list_dict_in_test[user_id])) + ' ' +\
                       ' '.join(user_like_list_dict_in_test[user_id]) + '\n'
                user_like_list_file.write(temp)
            for user_id in user_like_list_dict_in_cold_test.keys():
                temp = user_id + ' ' +\
                       str(len(user_like_list_dict_in_cold_test[user_id])) + ' ' +\
                       ' '.join(user_like_list_dict_in_cold_test[user_id]) + '\n'
                user_like_list_cold_file.write(temp)

            train_file.close()
            test_file.close()
            cold_test_file.close()
            user_like_list_file.close()
            user_like_list_cold_file.close()

    def generate_cross_validate_data(self):
        if self.__time_interval_days > 0:
            data_path = self.__data_path + 'data_divided_by_' +\
                                           str(self.__time_interval_days) +\
                                           '_days/'
        else:
            data_path = self.__data_path
        result_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(
            self.__filter_threshold)
        rating_data = np.loadtxt(result_path + '/rating_file.dat.txt',
                                 dtype=float)
        start_time = int(rating_data[0][3])
        end_time = int(rating_data[-1][3])

        # generate data at every time step
        for current_time_step in xrange(start_time, end_time + 1):
            time_path = result_path + '/time_step_' + str(current_time_step)
            if not os.path.isdir(time_path):
                os.mkdir(time_path)
            doc_liked_file = open(
                result_path + '/doc_liked_list_at_time_step' + str(
                    current_time_step) + '.dat.txt', 'r')
            ratings = []
            for line in doc_liked_file.readlines():
                splits = line.strip().split()
                doc_id = splits[0]
                for uid in xrange(2, len(splits)):
                    user_rating = splits[uid].split(':')
                    user_id = user_rating[0]
                    rating = user_rating[1]
                    ratings.append((user_id, doc_id, rating))
            self.cross_validate_on_ratings(ratings, time_path, current_time_step)
