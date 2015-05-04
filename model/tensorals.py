# encoding: utf-8
import os
import logging
import numpy as np

from utility import util
from utility import evaluate
from utility import fileutil

class Tensorals:
    filter_threshold = 10
    fold_num = 5
    model_name = 'tensor-als'

    def __init__(self, k=20, time_interval=360, times=0, dataset=''):
        self.k = k
        self.times = times
        self.time_interval = time_interval
        self.dataset = dataset
        self.data_path = \
            os.path.realpath(os.path.join(__file__,
                                          '../../data/preprocessed_data',
                                          'data_divided_by_' + str(time_interval) + '_days',
                                          'filtered_by_user_doc_like_list_len_' +\
                                          str(self.filter_threshold)))

        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(filename)s[line:%(lineno)d]\
                                    %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='./log/tensor-als_k_' +
                                     str(k) + '_timestep_' +
                                     str(time_interval) + '_' +
                                     str(self.times)  + '.log',
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

        user_id_map = np.loadtxt(os.path.join(self.data_path,
                                              'user_id_map.dat.txt'), int)
        doc_id_map = np.loadtxt(os.path.join(self.data_path,
                                             'doc_id_map.dat.txt'), int)

        user_time_dist = np.loadtxt(
            os.path.join(self.data_path, 'user_time_distribute.dat.txt'),
            int)
        doc_time_dist = np.loadtxt(
            os.path.join(self.data_path, 'doc_time_distribute.dat.txt'),
            int)

        user_time_dict = dict(zip(user_time_dist[:, 0], user_time_dist[:, 1]))
        doc_time_dict = dict(zip(doc_time_dist[:, 0], doc_time_dist[:, 1]))

        R = np.loadtxt(
            os.path.join(self.data_path, 'rating_file.dat.txt'),
            float)
        return (user_time_dict, doc_time_dict,
                user_id_map, doc_id_map, R)


    def evaluate(self, metric, metric_dict, predict_matrix, current_data_path,
                 recall_num, current_user_like_dict, cold = False):
        if cold:
            logging.info(str.format('\t{0}_for_cold at {1}:', metric, recall_num))
        else:
            logging.info(str.format('\t{0} at {1}:', metric, recall_num))
        metric_value = -1
        if metric == 'rmse':
            exec(str.format('metric_value = evaluate.get_{0}(predict_matrix,\
            current_data_path, cold)', metric))
        else:
            exec(str.format('metric_value = evaluate.get_{0}(predict_matrix,\
            current_data_path, recall_num, current_user_like_dict, cold)',
                            metric))
        if metric_value != -1:
            util.add_list_value_for_dict(metric_dict, recall_num, metric_value)
        logging.info('\t' + self.model_name + ' :  ' + str(metric_value) + '\n')

    def write_avg_metric_value(self, metric, metric_dict, recall_num,
                               metric_result_dir, cold=False):
        if cold:
            metric += '_for_cold_start'
            logging.info('for cold start evaluate...')
        logging.info(str.format('\tAverage {0} at {1}:', metric, recall_num))
        if recall_num not in metric_dict:
            logging.info('no test data!!!!!')
            avg_metric_value = -1
        else:
            avg_metric_value = util.avg_of_list(metric_dict[recall_num])
        logging.info('\t\tavg ' + self.model_name + ' :  ' +
                     str(avg_metric_value) + '\n\n\n')

        result_file_path = os.path.join(metric_result_dir,
                                        str.format('{0}_at_{1}.txt',
                                                   metric, recall_num))
        if os.path.isfile(result_file_path):
            result_file = open(result_file_path, 'a')
            result_file.write(str(avg_metric_value) + '\n')
        else:
            result_file = open(result_file_path, 'a')
            result_file.write(self.model_name + '\n')
            result_file.write(str(avg_metric_value) + '\n')

        result_file.close()

    def run(self):
        print 'k: %d\n' % self.k
        (user_time_dict, doc_time_dict,
         user_id_map, doc_id_map, R) = self.prepare_data()
        time_step_num = int(R[-1, 3])

        user_num = user_id_map.shape[0]
        doc_num = doc_id_map.shape[0]


        time_filter_dir = \
            os.path.realpath(os.path.join(__file__,
                                          '../../result/tensor-als_time_step_' +
                                          str(self.time_interval) +
                                          '_filter_by_' +
                                          str(self.filter_threshold)))
        fileutil.mkdir(time_filter_dir)

        result_dir = \
            os.path.join(
                time_filter_dir,
                str.format('fold_{0}_k_{1}_{2}_{3}', self.fold_num, self.k,
                           self.dataset, self.times))
        fileutil.mkdir(result_dir)

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

        start = 1

        # for all the consecutive periods
        for current_time_step in range(start + 1, time_step_num + 1):

            logging.info('\n=========================\n')
            logging.info('time_step number %i:\t' + str(current_time_step))
            logging.info('----------------\n')

            recall_dict = {}
            ndcg_dict = {}
            map_dict = {}
            rmse_dict = {}
            recall_cold_dict = {}
            ndcg_cold_dict = {}
            map_cold_dict = {}
            rmse_cold_dict = {}

            current_user_num = user_time_dict[current_time_step]
            current_doc_num = doc_time_dict[current_time_step]

            current_user_like_dict = {}
            like_file = open(os.path.join(self.data_path,
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
                current_data_path = \
                    os.path.join(self.data_path,
                                 str.format('time_step_{0}/data_{1}',
                                            current_time_step, fold_id))

                logging.info('begin to generate train and test file for tensor-als...\n')
                util.generate_train_and_test_file(
                                                  current_user_num,
                                                  current_doc_num,
                                                  current_data_path,
                                                  1,
                                                  current_time_step,
                                                  self.times,
                                                  'tensor-als')

                logging.info('\n\n begin training\n')

                tensor_exe_path = '../../../graphchi-cpp/toolkits/collaborative_filtering/als_tensor'
                tensor_exe_fullpath = os.path.realpath(os.path.join(__file__,
                                                                    tensor_exe_path))
                train_file_path = os.path.join(current_data_path, 'tensor-als_train' + str(self.times))
                test_file_path = os.path.join(current_data_path, 'tensor-als_test' + str(self.times))
                params = '--lambda=0.05 --minval=0 --maxval=1 --max_iter=30 --D=20 --quiet=1'
                command = '{0} --training={1} --test={2} {3}'.format(tensor_exe_fullpath,
                                                                     train_file_path,
                                                                     test_file_path,
                                                                     params)
                print command
                out = os.popen(command)
                str1 = out.read()
                while str1 != '':
                    print str1
                    str1 = out.read()

                logging.info('predict ratings\n')
                PredictR = util.create_predict_matrix(current_user_num,
                                                      current_doc_num,
                                                      current_data_path,
                                                      self.times,
                                                      'tensor-als')

                NormPR = PredictR / PredictR.max()

                logging.info('[ok]\n')

                logging.info('\t fold_id:' + str(fold_id) + '\n')
                for recall_num in [3, 10, 50, 100, 300, 500, 1000]:
                    # recall performance
                    self.evaluate('recall', recall_dict, NormPR,
                                  current_data_path, recall_num,
                                  current_user_like_dict)
                    # recall for cold start performance
                    self.evaluate('recall', recall_cold_dict, NormPR,
                                  current_data_path, recall_num,
                                  current_user_like_dict, cold=True)

                    # ndcg performance
                    self.evaluate('ndcg', ndcg_dict, NormPR,
                                  current_data_path, recall_num,
                                  current_user_like_dict)
                    # ndcg for cold start performance
                    self.evaluate('ndcg', ndcg_cold_dict, NormPR,
                                  current_data_path, recall_num,
                                  current_user_like_dict, cold=True)

                    # map performance
                    self.evaluate('map', map_dict, NormPR,
                                  current_data_path, recall_num,
                                  current_user_like_dict)
                    # map for cold start performance
                    self.evaluate('map', map_cold_dict, NormPR,
                                  current_data_path, recall_num,
                                  current_user_like_dict, cold=True)


                # rmse performance
                self.evaluate('rmse', rmse_dict, NormPR, current_data_path,
                              3, current_user_like_dict)
                # rmse for cold start performance
                self.evaluate('rmse', rmse_cold_dict, NormPR,
                              current_data_path, 3,
                              current_user_like_dict, cold=True)

            logging.info('current_time_step: ' + str(current_time_step) + '\n')

            for recall_num in [3, 10, 50, 100, 300, 500, 1000]:
                # recall
                self.write_avg_metric_value('recall', recall_dict, recall_num,
                                            recall_result_dir)
                self.write_avg_metric_value('recall', recall_cold_dict,
                                            recall_num, recall_result_dir,
                                            cold=True)
                # ndcg
                self.write_avg_metric_value('ndcg', ndcg_dict, recall_num,
                                            ndcg_result_dir)
                self.write_avg_metric_value('ndcg', ndcg_cold_dict, recall_num,
                                            ndcg_result_dir, cold=True)

                # map
                self.write_avg_metric_value('map', map_dict, recall_num,
                                            map_result_dir)
                self.write_avg_metric_value('map', map_cold_dict, recall_num,
                                            map_result_dir, cold=True)

            # rmse
            self.write_avg_metric_value('rmse', rmse_dict, 3, rmse_result_dir)
            self.write_avg_metric_value('rmse', rmse_cold_dict, 3,
                                        rmse_result_dir, cold=True)

        logging.info('\n all process done! exit now...')