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
k = 100

lambd = 0.5

eta = 0.2

if len(argvs) == 4:
    k = int(float(argvs[1]))
    lambd = float(argvs[2])
    eta = float(argvs[3])
    
print 'k: ', k, '\tlambda: ',  lambd, '\teta: ', eta ,'\n'

topic_num = 20
time_interval = 360
filter_threshold = 10

origin_data_path = './data/preprocessed_data/'
data_path = origin_data_path + 'data_divided_by_' + str(time_interval) + '_days/'
filter_data_path = data_path + 'filtered_by_user_doc_like_list_len_' + str(filter_threshold) +'/'

doc_id_map_after_filter = np.loadtxt(data_path + 'doc_id_citeulike_id_map_after_filter.dat.txt', int)
doc_id_map_after_filter_dict = dict(zip(doc_id_map_after_filter[:,0], \
        doc_id_map_after_filter[:,1]))

doc_id_map_in_total = open(origin_data_path + 'doc_id_citeulike_id_map.csv', 'r').readlines()
del doc_id_map_in_total[0]

doc_id_map_in_total_dict = {}
for row in doc_id_map_in_total:
    splits = row.strip().split(',')
    doc_id_map_in_total_dict[int(splits[1])] = int(splits[0])

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

doc_id_citeulike_id_dict = {}
X = np.zeros((doc_id_map.shape[0], 8000))

doc_words = open(origin_data_path + 'mult.dat.txt', 'r').readlines()

# get doc_id citeulike_id dict and doc word matrix
for i in range(doc_id_map.shape[0]):
    citeulike_id = doc_id_map_after_filter_dict[doc_id_map[i, 1]]
    doc_id_citeulike_id_dict[doc_id_map[i, 0]] = citeulike_id

    doc_id_in_total = doc_id_map_in_total_dict[citeulike_id]
    words = doc_words[doc_id_in_total-1].strip().split()
    del words[0]
    for w in words:
        splits = w.split(':')
        X[doc_id_map[i, 0]-1, int(splits[0])] = int(splits[1])

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

regl1nmf = 0.05

regl1jpp = 0.05

epsilon = 1

maxiter = 100

#recall_num = 100

fold_num = 5


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d]\
                            %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/ttarm_k_' + str(k) + '_lambda_' + \
                            str(lambd) + '_alpha_' + str(regl1jpp) + '_eta_' + str(eta) + '.log',
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

result_dir = time_filter_dir + '/ttarm_eta_' + str(eta) + '_fold_' + str(fold_num) + \
        '_k_' + str(k) + '_lambda_' + str(lambd) + '_alpha_' + str(regl1jpp)
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
logging.info('start nmf:\n')
(P, Q) = util.nmf(Rt, k, maxiter, regl1nmf, epsilon)
logging.info('[ok]\n')

logging.info(str(P.shape) + '\t' + str(Q.shape) + '\n')

# init W and H with nmf
Xt = X[range(doc_time_dict[start]), :]
(W1, H1) = util.nmf(Xt, topic_num, maxiter, regl1nmf, epsilon)

# learning topic distribution by jpp
logging.info('start learning topic distribution by jpp ......')

(W, H, M) = JPPTopic(X, H1, topic_num, lambd, regl1jpp, epsilon, 100, True)

logging.info('end')

# number of period we consider
finT = time_step_num

#for all the consecutive periods
for current_time_step in range(start+1, finT+1):

    logging.info('\n=========================\n')
    logging.info('time_step number %i:\t' + str(current_time_step))
    logging.info('----------------\n')

    if current_time_step == start+1:
        Po2 = P
    else:
        Po2 = P2

    j_topic_recall_dict = {}
    j_topic_ndcg_dict = {}
    j_topic_ap_dict = {}

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
        current_data_path = filter_data_path + 'time_step_' + \
                            str(current_time_step) + '/data_' + \
                            str(fold_id)
        train_data_path = current_data_path + '/train.dat.txt'
        
        Rt = util.generate_matrice_for_file2(train_data_path,
                                             current_user_num,
                                             current_doc_num,
                                             ruser_id_dict,
                                             rdoc_id_dict
                                             )
        logging.info('non zero cell num: ' + str(len(np.nonzero(Rt)[0])))

        # calculate user item topic similarity matrix
        Ct_train = util.cal_topic_similarity_matrix(W,
                                              current_data_path,
                                              current_user_num,
                                              current_doc_num,
                                              ruser_id_dict,
                                              rdoc_id_dict,
                                              current_user_like_dict,
                                              True)

        logging.info('computing JPP_with_topic decomposition...')

        Po2 = util.reshape_matrix(Po2, current_user_num, k)

        P2, Q2, S2 = JPPCF_with_topic(Rt, Po2, Ct_train, k, eta, lambd, regl1jpp,  epsilon, maxiter, True)

        Ct_test = util.cal_topic_similarity_matrix(W,
                                              current_data_path,
                                              current_user_num,
                                              current_doc_num,
                                              ruser_id_dict,
                                              rdoc_id_dict,
                                              current_user_like_dict,
                                              False)
        PredictR2 = ((1-eta)*np.dot(P2, Q2)) + (eta*Ct_test)
        NormPR2 = PredictR2 / PredictR2.max()

        logging.info('[ok]\n')

        logging.info('\t fold_id:' + str(fold_id) + '\n')
        for recall_num in [3,5,10,20,50,100,150,200,250,300]:
            logging.info('\trecall at ' + str(recall_num) + ':')

            jppcf_with_topic_recall = util.performance_cross_validate_recall2(
                    NormPR2, current_data_path, recall_num,
                    ruser_id_dict, rdoc_id_dict, current_user_like_dict)

            if j_topic_recall_dict.has_key(recall_num):
                    j_topic_recall_dict[recall_num].append(jppcf_with_topic_recall)
            else:
                    j_topic_recall_dict[recall_num] = [jppcf_with_topic_recall]
            logging.info('\tJPPCF_with_topic :  ' + str(jppcf_with_topic_recall) + '\n')

            # ndcg performance
            logging.info('\nndcg at ' + str(recall_num) + ':')
            jppcf_with_topic_ndcg = util.performance_ndcg(
                    NormPR2, current_data_path, recall_num,
                    ruser_id_dict, rdoc_id_dict, current_user_like_dict)

            if j_topic_ndcg_dict.has_key(recall_num):
                    j_topic_ndcg_dict[recall_num].append(jppcf_with_topic_ndcg)
            else:
                    j_topic_ndcg_dict[recall_num] = [jppcf_with_topic_ndcg]
            logging.info('\tJPPCF_with_topic :  ' + str(jppcf_with_topic_ndcg) + '\n')

            # ap performance
            logging.info('\nap at ' + str(recall_num) + ':')
            jppcf_with_topic_ap = util.performance_ndcg(
                    NormPR2, current_data_path, recall_num,
                    ruser_id_dict, rdoc_id_dict, current_user_like_dict)
            if j_topic_ap_dict.has_key(recall_num):
                    j_topic_ap_dict[recall_num].append(jppcf_with_topic_ap)
            else:
                    j_topic_ap_dict[recall_num] = [jppcf_with_topic_ap]
            logging.info('\tJPPCF_with_topic :  ' + str(jppcf_with_topic_ap) + '\n')

    logging.info('current_time_step: ' + str(current_time_step) + '\n')

    for recall_num in [3,5,10,20,50,100,150,200,250,300]:
        # recall
        logging.info('\trecall at ' + str(recall_num) + ':')
        avg_jppcf_with_topic_recall = util.avg_of_list(j_topic_recall_dict[recall_num])
        logging.info('\t\tavg JPPCF_with_topic :  ' + str(avg_jppcf_with_topic_recall) + '\n\n\n')

        exist = False
        if os.path.isfile(recall_result_dir + '/recall_at_' + str(recall_num)  + '.txt'):
            exist = True

        result_file = open(recall_result_dir + '/recall_at_' + str(recall_num)  + '.txt', 'a')
        if not exist:
            result_file.write('jppcf_with_topic\n')

        result_file.write(str(avg_jppcf_with_topic_recall) + '\n')
        result_file.close()

        # ndcg
        logging.info('\tndcg at ' + str(recall_num) + ':')
        avg_jppcf_with_topic_ndcg = util.avg_of_list(j_topic_ndcg_dict[recall_num])
        logging.info('\t\tavg JPPCF_with_topic :  ' + str(avg_jppcf_with_topic_ndcg) + '\n\n\n')

        exist = False
        if os.path.isfile(ndcg_result_dir + '/ndcg_at_' + str(recall_num)  + '.txt'):
            exist = True

        result_file = open(ndcg_result_dir + '/ndcg_at_' + str(recall_num)  + '.txt', 'a')
        if not exist:
            result_file.write('jppcf_with_topic\n')

        result_file.write(str(avg_jppcf_with_topic_ndcg) + '\n')
        result_file.close()
        
        # ap
        logging.info('\tap at ' + str(recall_num) + ':')
        avg_jppcf_with_topic_ap = util.avg_of_list(j_topic_ap_dict[recall_num])
        logging.info('\t\tavg JPPCF_with_topic :  ' + str(avg_jppcf_with_topic_ap) + '\n\n\n')

        exist = False
        if os.path.isfile(ap_result_dir + '/ap_at_' + str(recall_num)  + '.txt'):
            exist = True

        result_file = open(ap_result_dir + '/ap_at_' + str(recall_num)  + '.txt', 'a')
        if not exist:
            result_file.write('jppcf_with_topic\n')

        result_file.write(str(avg_jppcf_with_topic_ap) + '\n')
        result_file.close()

    logging.info('=========================\n')

logging.info('\n all process done! exit now...')
