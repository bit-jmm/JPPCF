# encoding: utf-8
import os
import logging

from utility import util
from utility import evaluate
from utility import fileutil
from model.JPPCF import *


class Ttarm:
    topic_num = 20
    time_interval = 360
    filter_threshold = 10
    regl1nmf = 0.05
    regl1jpp = 0.05
    epsilon = 1
    maxiter = 100
    fold_num = 5
    model_name = 'TTARM'

    def __init__(self, k=20, lambd=10, eta=0.3):
        self.k = k
        self.lambd = lambd
        self.eta = eta
        self.origin_data_path = \
            os.path.normpath(os.path.join(__file__,
                                          '../../data/preprocessed_data'))

        self.data_path = os.path.join(self.origin_data_path,
                                      'data_divided_by_' +
                                      str(self.time_interval) + '_days')
        self.filter_data_path = \
            os.path.join(self.data_path,
                         'filtered_by_user_doc_like_list_len_' +
                         str(self.filter_threshold))

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d]\
                                    %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='../log/new_ttarm_k_' +
                                     str(k) + '_lambda_' +
                                     str(lambd) + '_alpha_' +
                                     str(self.regl1jpp) + '_eta_' +
                                     str(eta) + '.log',
                            filemode='w')

        ##################################################################
        # 定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，
        # 并将其添加到当前的日志处理对象#
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        ##################################################################

    def prepare_data(self):

        user_id_map = np.loadtxt(os.path.join(self.filter_data_path,
                                              'user_id_map.dat.txt'), int)
        doc_id_map = np.loadtxt(os.path.join(self.filter_data_path,
                                             'doc_id_map.dat.txt'), int)

        user_time_dist = np.loadtxt(
            os.path.join(self.filter_data_path, 'user_time_distribute.dat.txt'),
            int)
        doc_time_dist = np.loadtxt(
            os.path.join(self.filter_data_path, 'doc_time_distribute.dat.txt'),
            int)

        user_time_dict = dict(zip(user_time_dist[:, 0], user_time_dist[:, 1]))
        doc_time_dict = dict(zip(doc_time_dist[:, 0], doc_time_dist[:, 1]))

        R = np.loadtxt(
            os.path.join(self.filter_data_path, 'rating_file.dat.txt'),
            float)
        return (user_time_dict, doc_time_dict,
                user_id_map, doc_id_map, R)

    def topic_distribute(self, doc_id_map, doc_time_dict):
        doc_id_map_after_filter = \
            np.loadtxt(
                os.path.join(self.filter_data_path,
                             'doc_id_citeulike_id_map_after_filter.dat.txt'),
                int)

        doc_id_citeulike_id_map = open(self.origin_data_path +
                                       '/doc_id_citeulike_id_map.csv',
                                       'r').readlines()
        del doc_id_citeulike_id_map[0]

        citeulike_id_doc_id_dict = {}
        for row in doc_id_citeulike_id_map:
            splits = row.strip().split(',')
            citeulike_id_doc_id_dict[int(splits[1])] = int(splits[0])

        doc_id_citeulike_id_map_after_filter_dict = dict(
            zip(doc_id_map_after_filter[:, 0], doc_id_map_after_filter[:, 1]))

        X = np.zeros((doc_id_map.shape[0], 8000))

        doc_words = open(os.path.join(self.origin_data_path, 'mult.dat.txt'),
                         'r').readlines()

        # get doc_id citeulike_id dict and doc word matrix
        for i in xrange(doc_id_map.shape[0]):
            citeulike_id = doc_id_citeulike_id_map_after_filter_dict[
                doc_id_map[i, 1]]

            doc_id_in_total = citeulike_id_doc_id_dict[citeulike_id]
            words = doc_words[doc_id_in_total - 1].strip().split()
            del words[0]
            for w in words:
                splits = w.split(':')
                X[doc_id_map[i, 1], int(splits[0])] = int(splits[1])

        # init W and H with nmf
        Xt = X[range(doc_time_dict[1]), :]
        (W1, H1) = util.nmf(Xt, self.topic_num, self.maxiter,
                            self.regl1nmf, self.epsilon)

        # learning topic distribution by jpp
        logging.info('start learning topic distribution by jpp ......')

        (W, H, M) = JPPTopic(X, H1, self.topic_num, self.lambd,
                             self.regl1jpp, self.epsilon, self.maxiter, True)

        logging.info('end')
        return W

    def evaluate(self, metric, metric_dict, predict_matrix, current_data_path,
                 recall_num, current_user_like_dict):
        logging.info(str.format('\t{0} at {1}:', metric, recall_num))
        metric_value = None
        if metric == 'rmse':
            exec(str.format('metric_value = evaluate.get_{0}(predict_matrix,\
            current_data_path)', metric))
        else:
            exec(str.format('metric_value = evaluate.get_{0}(predict_matrix,\
            current_data_path, recall_num, current_user_like_dict)', metric))
        util.add_list_value_for_dict(metric_dict, recall_num, metric_value)
        logging.info('\t' + self.model_name + ' :  ' + str(metric_value) + '\n')

    def write_avg_metric_value(self, metric, metric_dict, recall_num,
                               metric_result_dir):
        logging.info(str.format('\tAverage {0} at {1}:', metric, recall_num))

        avg_metric_value = util.avg_of_list(metric_dict[recall_num])
        logging.info('\t\tavg ' + self.model_name + ' :  ' +
                     str(avg_metric_value) + '\n\n\n')

        exist = False
        if os.path.isfile(metric_result_dir + '/' + metric + '_at_' + str(
                recall_num) + '.txt'):
            exist = True
        result_file = open(metric_result_dir + '/' + metric + '_at_' + str(
                           recall_num) + '.txt', 'a')
        if not exist:
            result_file.write(self.model_name+ '\n')

        result_file.write(str(avg_metric_value) + '\n')
        result_file.close()

    def run(self):
        print 'k: %d\tlambda:%d \teta: %.1f\n' % (self.k, self.lambd, self.eta)
        (user_time_dict, doc_time_dict,
         user_id_map, doc_id_map, R) = self.prepare_data()
        time_step_num = int(R[-1, 3])

        user_num = user_id_map.shape[0]
        doc_num = doc_id_map.shape[0]

        W = self.topic_distribute(doc_id_map, doc_time_dict)

        time_filter_dir = \
            os.path.normpath(os.path.join(__file__,
                                          '../../result/new_ttarm_time_step_' +
                                          str(self.time_interval) +
                                          '_filter_by_' +
                                          str(self.filter_threshold)))
        fileutil.mkdir(time_filter_dir)

        result_dir = os.path.join(time_filter_dir,
                                  '/eta_' + str(self.eta) +
                                  '_fold_' + str(self.fold_num) +
                                  '_k_' + str(self.k) + '_lambda_' +
                                  str(self.lambd) + '_alpha_' +
                                  str(self.regl1jpp))

        recall_result_dir = os.path.join(result_dir, 'recall')
        ndcg_result_dir = os.path.join(result_dir, 'ndcg')
        map_result_dir = os.path.join(result_dir, 'map')
        rmse_result_dir = os.path.join(result_dir, 'rmse')

        for d in [recall_result_dir, ndcg_result_dir,
                  map_result_dir, rmse_result_dir]:
            fileutil.mkdir(d)

        logging.info('user num: ' + str(user_num) + '\n')
        logging.info('doc num: ' + str(doc_num) + '\n')
        logging.info('time step num: ' + str(time_step_num) + '\n')

        # the start time period used for init of W(1) and H(1), using normal NMF
        start = 1
        Rt = util.generate_matrice_between_time(R, user_time_dict[start],
                                                doc_time_dict[start], start,
                                                start)

        logging.info('non zero cell num: ' + str(len(np.nonzero(Rt)[0])))
        logging.info('start nmf:\n')

        (P, Q) = util.nmf(Rt, self.k, self.maxiter, self.regl1nmf, self.epsilon)
        logging.info('[ok]\n')

        # for all the consecutive periods
        for current_time_step in range(start + 1, time_step_num + 1):

            logging.info('\n=========================\n')
            logging.info('time_step number %i:\t' + str(current_time_step))
            logging.info('----------------\n')

            if current_time_step == start + 1:
                Po2 = P
            else:
                Po2 = P2

            recall_dict = {}
            ndcg_dict = {}
            map_dict = {}
            rmse_dict = {}

            current_user_num = user_time_dict[current_time_step]
            current_doc_num = doc_time_dict[current_time_step]

            current_user_like_dict = {}
            like_file = open(os.path.join(self.filter_data_path,
                                          'user_like_list_at_time_step' +
                                          str(current_time_step) + '.dat.txt'))
            for line in like_file.readlines():
                splits = line.strip().split()
                like_list = []
                for i in xrange(2, len(splits)):
                    doc_rating = splits[i].split(':')
                    like_list.append((doc_rating[0], doc_rating[1]))
                current_user_like_dict[splits[0]] = like_list

            for fold_id in range(self.fold_num):
                current_data_path = self.filter_data_path + 'time_step_' + \
                                    str(current_time_step) + '/data_' + \
                                    str(fold_id)
                train_data_path = current_data_path + '/train.dat.txt'

                Rt = util.generate_matrice_for_file(train_data_path,
                                                    current_user_num,
                                                    current_doc_num
                                                    )
                logging.info('non zero cell num: ' + str(len(np.nonzero(Rt)[0])))

                # calculate user item topic similarity matrix
                Ct_train = \
                    util.cal_topic_similarity_matrix(W,
                                                     current_data_path,
                                                     current_user_num,
                                                     current_doc_num,
                                                     current_user_like_dict,
                                                     True)
                logging.info('computing ' + self.model_name +
                             ' decomposition...')

                Po2 = util.reshape_matrix(Po2, current_user_num, self.k)

                P2, Q2, S2 = JPPCF_with_topic(Rt, Po2, Ct_train, self.k,
                                              self.eta,
                                              self.lambd, self.regl1jpp,
                                              self.epsilon, self.maxiter, True)

                Ct_test = \
                    util.cal_topic_similarity_matrix(W,
                                                     current_data_path,
                                                     current_user_num,
                                                     current_doc_num,
                                                     current_user_like_dict,
                                                     False)

                PredictR2 = ((1 - self.eta) * np.dot(P2, Q2)) + \
                            (self.eta * Ct_test)
                NormPR2 = PredictR2 / PredictR2.max()

                logging.info('[ok]\n')

                logging.info('\t fold_id:' + str(fold_id) + '\n')
                for recall_num in [3, 10, 50, 100, 300, 500, 1000]:
                    # recall evaluate
                    self.evaluate('recall', recall_dict, NormPR2,
                                  current_data_path, recall_num,
                                  current_user_like_dict)
                    # ndcg performance
                    self.evaluate('ndcg', ndcg_dict, NormPR2,
                                  current_data_path, recall_num,
                                  current_user_like_dict)
                    # map performance
                    self.evaluate('map', map_dict, NormPR2,
                                  current_data_path, recall_num,
                                  current_user_like_dict)
                    # rmse performance
                    self.evaluate('rmse', rmse_dict, NormPR2,
                                  current_data_path, recall_num,
                                  current_user_like_dict)

            logging.info('current_time_step: ' + str(current_time_step) + '\n')

            for recall_num in [3, 10, 50, 100, 300, 500, 1000]:
                # recall
                self.write_avg_metric_value('recall', recall_dict, recall_num,
                                            recall_result_dir)
                # ndcg
                self.write_avg_metric_value('ndcg', ndcg_dict, recall_num,
                                            ndcg_result_dir)
                # map
                self.write_avg_metric_value('map', map_dict, recall_num,
                                            map_result_dir)
                # rmse
                self.write_avg_metric_value('rmse', rmse_dict, recall_num,
                                            rmse_result_dir)

        logging.info('\n all process done! exit now...')
