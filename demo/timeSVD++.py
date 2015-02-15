# encoding: utf-8
import sys
sys.path.append('/home/zjd/jmm/JPPCF/')

import os
import numpy as np
import util
from JPPCF import *

import logging

argvs = sys.argv

# We fix the num of latent feature
k = 200

if len(argvs) == 2:
    k = int(float(argvs[1]))
    
print 'k: ', k, '\n'

time_interval = 360
filter_threshold = 10

origin_data_path = './data/preprocessed_data/'
data_path = origin_data_path + 'data_divided_by_' + str(time_interval) + '_days/'
filter_data_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(filter_threshold) +'/'

user_id_map = np.loadtxt(filter_data_path + 'user_id_map.dat.txt', int)
doc_id_map = np.loadtxt(filter_data_path + 'doc_id_map.dat.txt', int)
user_time_dist = np.loadtxt(filter_data_path + 'user_time_distribute.dat.txt', int)
doc_time_dist = np.loadtxt(filter_data_path + 'doc_time_distribute.dat.txt', int)


user_time_dict = dict(zip(user_time_dist[:,0], user_time_dist[:,1]))
doc_time_dict = dict(zip(doc_time_dist[:,0], doc_time_dist[:,1]))

user_id_dict = dict(zip(user_id_map[:,0], user_id_map[:,1]))
ruser_id_dict = dict(zip(user_id_map[:,1], user_id_map[:,0]))
doc_id_dict = dict(zip(doc_id_map[:,0], doc_id_map[:,1]))
rdoc_id_dict = dict(zip(doc_id_map[:,1], doc_id_map[:,0]))

user_num = user_id_map.shape[0]
doc_num = doc_id_map.shape[0]
print 'user_num: ', user_num, '\n'
print 'doc_num: ', doc_num, '\n'

R = np.loadtxt(filter_data_path + 'rating_file.dat.txt', int)
time_step_num = R[:, 2].max()

row = R.shape[0]
for i in range(row):
    user_id = R[i, 0]
    doc_id = R[i, 1]
    R[i, 0] = ruser_id_dict[user_id]
    R[i, 1] = rdoc_id_dict[doc_id]

#exit(0)

#recall_num = 100

fold_num = 1


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d]\
                            %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/timeSVDpp_k_' + str(k) + '.log',
                    filemode='w')

##################################################################
#定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，
#并将其添加到当前的日志处理对象#
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
##################################################################
#logging.debug('This is debug message')
#logging.info('This is info message')
#logging.warning('This is warning message')


time_filter_dir = './result/time_interval_' + str(time_interval) + '_filter_by_' + \
                  str(filter_threshold)
if not os.path.isdir(time_filter_dir):
    os.mkdir(time_filter_dir)

result_dir = time_filter_dir + '/timesvdpp_fold_' + str(fold_num) + '_k_' + str(k)
recall_result_dir = result_dir + '/recall'
ndcg_result_dir = result_dir + '/ndcg'
ap_result_dir = result_dir + '/ap'

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
if not os.path.isdir(recall_result_dir):
    os.mkdir(recall_result_dir)
if not os.path.isdir(ndcg_result_dir):
    os.mkdir(ndcg_result_dir)
if not os.path.isdir(ap_result_dir):
    os.mkdir(ap_result_dir)  

logging.info('user num: ' + str(user_num) + '\n')
logging.info('doc num: ' + str(doc_num) + '\n')
logging.info('time step num: ' + str(time_step_num) + '\n')

# the start time period used for init of W(1) and H(1), using normal NMF
start = 1
Rt = util.generate_matrice_between_time(R, user_time_dict[start], doc_time_dict[start], start, start)

logging.info('non zero cell num: ' + str(len(np.nonzero(Rt)[0])))

# number of period we consider
finT = time_step_num

#for all the consecutive periods
for current_time_step in range(start+1, finT+1):

    logging.info('\n=========================\n')
    logging.info('time_step number %i:\t' + str(current_time_step))
    logging.info('----------------\n')

    recall_dict = {}

    ndcg_dict = {}

    ap_dict = {}

    current_user_num = user_time_dict[current_time_step]
    current_doc_num = doc_time_dict[current_time_step]

    current_user_like_dict = {}
    like_file = open(filter_data_path + 'user_like_list_at_time_step' + \
                     str(current_time_step) + '.dat.txt')
    for user in like_file.readlines():
        splits = user.split()
        like_list = []
        for i in range(2, len(splits)):
            like_list.append(rdoc_id_dict[int(splits[i])])
        current_user_like_dict[ruser_id_dict[int(splits[0])]] = like_list

    for fold_id in range(fold_num):
        logging.info('\n\n fold_id:' + str(fold_id) + '\n\n')
        current_data_path = filter_data_path + 'time_step_' + \
                            str(current_time_step) + '/data_' + \
                            str(fold_id)
        train_data_path = current_data_path + '/train.dat.txt'
        
        logging.info('non zero cell num: ' + str(len(np.nonzero(Rt)[0])))
        logging.info('begin to generate train and test file for timeSVD++...\n')
        util.generate_train_and_test_file_for_timesvdpp(R,
                                                current_user_num,
                                                current_doc_num,
                                                current_data_path,
                                                ruser_id_dict,
                                                rdoc_id_dict,
                                                start + 1,
                                                current_time_step-1)
        logging.info('end.\n')
        
        logging.info('\n\n begin training and predict\n')
        time_train_path = '/home/zjd/jmm/JPPCF/data/preprocessed_data/data_divided_by_360_days/filtered_by_user_doc_like_list_len_10/time_step_' + str(current_time_step) + '/data_' + str(fold_id)
        out = os.popen('../graphchi-cpp/toolkits/collaborative_filtering/timesvdpp --training=' + time_train_path + '/timesvdpp_train' + ' --test=' + time_train_path + '/timesvdpp_test' + ' --minval=0 --maxval=1 --max_iter=50 --quiet=1 --D=200')
        str1 = out.read()
        while str1 != '':
            print str1
            str1 = out.read()
        NormPR = util.create_predict_matrix(current_user_num, current_doc_num, current_data_path)
        
        logging.info('[ok]\n')
        
        for recall_num in [3,5,10,20,50,100,150,200,250,300]:
            logging.info('\trecall at ' + str(recall_num) + ':')
            timesvdpp_recall = util.performance_cross_validate_recall2(
                    NormPR, current_data_path, recall_num,
                    ruser_id_dict, rdoc_id_dict, current_user_like_dict)

            if recall_dict.has_key(recall_num):
                    recall_dict[recall_num].append(timesvdpp_recall)
            else:
                    recall_dict[recall_num] = [timesvdpp_recall]
                    
            logging.info('\t\ttimeSVD++ :  ' + str(timesvdpp_recall) + '\n')

            # ndcg performance
            logging.info('\nndcg at ' + str(recall_num) + ':')
            timesvdpp_ndcg = util.performance_ndcg(
                    NormPR, current_data_path, recall_num,
                    ruser_id_dict, rdoc_id_dict, current_user_like_dict)

            if ndcg_dict.has_key(recall_num):
                    ndcg_dict[recall_num].append(timesvdpp_ndcg)
            else:
                    ndcg_dict[recall_num] = [timesvdpp_ndcg]

            logging.info('\t\ttimeSVD++ :  ' + str(timesvdpp_ndcg) + '\n')

            # ap performance
            logging.info('\nap at ' + str(recall_num) + ':')
            timesvdpp_ap = util.performance_ap(
                    NormPR, current_data_path, recall_num,
                    ruser_id_dict, rdoc_id_dict, current_user_like_dict)

            if ap_dict.has_key(recall_num):
                    ap_dict[recall_num].append(timesvdpp_ap)
            else:
                    ap_dict[recall_num] = [timesvdpp_ap]
                    
            logging.info('\t\ttimeSVD++ :  ' + str(timesvdpp_ap) + '\n')

    logging.info('current_time_step: ' + str(current_time_step) + '\n')

    for recall_num in [3,5,10,20,50,100,150,200,250,300]:
        # recall
        logging.info('\trecall at ' + str(recall_num) + ':')
        avg_timesvdpp_recall = util.avg_of_list(recall_dict[recall_num])
        logging.info('\t\tavg timeSVD++ :  ' + str(avg_timesvdpp_recall) + '\n')

        exist = False
        if os.path.isfile(recall_result_dir + '/recall_at_' + str(recall_num)  + '.txt'):
            exist = True

        result_file = open(recall_result_dir + '/recall_at_' + str(recall_num)  + '.txt', 'a')
        if not exist:
            result_file.write('timeSVD++\n')

        result_file.write(str(avg_timesvdpp_recall) + '\n')
        result_file.close()

        # ndcg
        logging.info('\tndcg at ' + str(recall_num) + ':')
        avg_timesvdpp_ndcg = util.avg_of_list(ndcg_dict[recall_num])
        logging.info('\t\tavg timeSVD++ :  ' + str(avg_timesvdpp_ndcg) + '\n')

        exist = False
        if os.path.isfile(ndcg_result_dir + '/ndcg_at_' + str(recall_num)  + '.txt'):
            exist = True

        result_file = open(ndcg_result_dir + '/ndcg_at_' + str(recall_num)  + '.txt', 'a')
        if not exist:
            result_file.write('timeSVD++\n')

        result_file.write(str(avg_timesvdpp_ndcg) + '\n')
        result_file.close()
        
        # ap
        logging.info('\tap at ' + str(recall_num) + ':')
        avg_timesvdpp_ap = util.avg_of_list(ap_dict[recall_num])
        logging.info('\t\tavg timeSVD++ :  ' + str(avg_timesvdpp_ap) + '\n')

        exist = False
        if os.path.isfile(ap_result_dir + '/ap_at_' + str(recall_num)  + '.txt'):
            exist = True

        result_file = open(ap_result_dir + '/ap_at_' + str(recall_num)  + '.txt', 'a')
        if not exist:
            result_file.write('timeSVD++\n')

        result_file.write(str(avg_timesvdpp_ap) + '\n')
        result_file.close()

    logging.info('=========================\n')

logging.info('\n all process done! exit now...')
