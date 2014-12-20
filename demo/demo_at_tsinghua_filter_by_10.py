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

if len(argvs) == 3:
    k = int(float(argvs[1]))
    lambd = float(argvs[2])
print 'k: ', k, '\tlambda: ',  lambd, '\n'

data_path = './data/preprocessed_data/filtered_by_user_doc_like_list_len_10/'
R = np.loadtxt(data_path + 'rating_file.dat.txt', int)
user_num = R[:, 0].max()
doc_num = R[:, 1].max()
time_step_num = R[:, 2].max()

print 'user_num: ', user_num, '\n'
print 'doc_num: ', doc_num, '\n'

#exit(0)



regl1nmf = 0.005

regl1jpp = 0.05

epsilon = 1

maxiter = 70

#recall_num = 100

fold_num = 1

Rall = util.generate_matrice_between_time(R, user_num, doc_num, 1, time_step_num)
#Rall = np.ones((100, 50))

print Rall.shape


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d]\
                            %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/tsinghua_server/filter_by_10/k_' + str(k) + '_lambda_' + \
                            str(lambd) + '_alpha_' + str(regl1jpp) + '.log',
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


result_dir = './result/tsinghua_server/filter_by_10/cross_validate_fold_' + str(fold_num) + \
        '_3models_k_' + str(k) + '_lambda_' + str(lambd) + '_alpha_' + str(regl1jpp)
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

logging.info('user num: ' + str(user_num) + '\n')
logging.info('doc num: ' + str(doc_num) + '\n')
logging.info('time step num: ' + str(time_step_num) + '\n')

# the start time period used for init of W(1) and H(1), using normal NMF
start = 1
Rt = util.generate_matrice_between_time(R, user_num, doc_num, start, start)

#Rt = np.ones((100, 50))
logging.info('non zero cell num: ' + str(len(np.nonzero(Rt)[0])))
logging.info('start nmf:\n')
(P, Q) = util.nmf(Rt, k, maxiter, regl1nmf, epsilon)
print P.shape, Q.shape

# number of period we consider
finT = time_step_num - 1

#for all the consecutive periods
for current_time_step in range(start+1, finT + 1):

    logging.info('\n=========================\n')
    logging.info('time_step number %i:\t' + str(current_time_step))
    logging.info('----------------\n')

    #Rtall = util.generate_matrice_between_time(R, user_num, doc_num,
    #                            current_time_step, current_time_step)
    Po = P

    trecall_dict = {}
    frecall_dict = {}
    jrecall_dict = {}

    for fold_id in range(fold_num):
    #for fold_id in [0]:
        train_data_path = data_path + 'time_step_' + str(current_time_step) + \
                '/data_' + str(fold_id) + '/train.dat.txt'
        Rt = util.generate_matrice_for_file(train_data_path, user_num, doc_num)
        logging.info('non zero cell num: ' + str(len(np.nonzero(Rt)[0])))
        #Rt = np.ones((100, 50))

        logging.info('computing JPP decomposition...')
        P, Q, S = JPPCF(Rt, Po, Po.shape[1], lambd, regl1jpp,  epsilon, maxiter, True)
        PredictR = np.dot(P, Q)
        NormPR = PredictR / PredictR.max()


        logging.info('[ok]\ncomputing t-model NMF decomposition...')
        Pbaseline, Qbaseline = util.nmf(Rt, k, maxiter, regl1nmf, epsilon)

        PredictRbaseline = np.dot(Pbaseline, Qbaseline)
        NormPRbaseline = PredictRbaseline / PredictRbaseline.max()

        logging.info('[ok]\ncomputing fix_model NMF decomposition...')
        Rt = util.generate_matrice_between_time(R,
                                                user_num,
                                                doc_num,
                                                start + 1,
                                                current_time_step-1,
                                                train_data_path)
        #Rt = np.ones((100, 50))
        logging.info('non zero cell num: ' + str(len(np.nonzero(Rt)[0])))
        Pbaseline2, Qbaseline2 = util.nmf(Rt, k, maxiter, regl1nmf, epsilon)

        PredictRbaseline2 = np.dot(Pbaseline2, Qbaseline2)
        NormPRbaseline2 = PredictRbaseline2 / PredictRbaseline2.max()

        logging.info('[ok]\n')

        logging.info('\t fold_id:' + str(fold_id) + '\n')
        for recall_num in [10,50,100,300,500,1000]:
            logging.info('\trecall at ' + str(recall_num) + ':')
            current_data_path = data_path + 'time_step_' + \
                                str(current_time_step) + '/data_' + \
                                str(fold_id)
            tnmf_recall = util.performance_cross_validate_recall(
                    NormPRbaseline, current_data_path, recall_num)
            fnmf_recall = util.performance_cross_validate_recall(
                    NormPRbaseline2, current_data_path, recall_num)
            jppcf_recall = util.performance_cross_validate_recall(
                    NormPR, current_data_path, recall_num)

            if trecall_dict.has_key(recall_num):
                    trecall_dict[recall_num].append(tnmf_recall)
            else:
                    trecall_dict[recall_num] = [tnmf_recall]

            if frecall_dict.has_key(recall_num):
                    frecall_dict[recall_num].append(fnmf_recall)
            else:
                    frecall_dict[recall_num] = [fnmf_recall]

            if jrecall_dict.has_key(recall_num):
                    jrecall_dict[recall_num].append(jppcf_recall)
            else:
                    jrecall_dict[recall_num] = [jppcf_recall]
            logging.info('\t\tt-model NMF :  ' + str(tnmf_recall) + '\n')
            logging.info('\t\tf-model NMF :  ' + str(fnmf_recall) + '\n')
            logging.info('\t\tJPPCF :  ' + str(jppcf_recall) + '\n')

    logging.info('current_time_step: ' + str(current_time_step) + '\n')

    for recall_num in [10,50,100,300,500,1000]:
        logging.info('\trecall at ' + str(recall_num) + ':')
        avg_tnmf_recall = util.avg_of_list(trecall_dict[recall_num])
        avg_fnmf_recall = util.avg_of_list(frecall_dict[recall_num])
        avg_jppcf_recall = util.avg_of_list(jrecall_dict[recall_num])
        logging.info('\t\tavg t-model NMF :  ' + str(avg_tnmf_recall) + '\n')
        logging.info('\t\tavg f-model NMF :  ' + str(avg_fnmf_recall) + '\n')
        logging.info('\t\tavg JPPCF :  ' + str(avg_jppcf_recall) + '\n')

        exist = False
        if os.path.isfile(result_dir + '/recall_at_' + str(recall_num)  + '.txt'):
            exist = True

        result_file = open(result_dir + '/recall_at_' + str(recall_num)  + '.txt', 'a')
        if not exist:
            result_file.write('t-model-nmf\tfix-model-nmf\tjppcf\n')

        result_file.write(str(avg_tnmf_recall) + '\t' + str(avg_fnmf_recall) + \
                '\t' + str(avg_jppcf_recall) + '\n')
        result_file.close()

    logging.info('=========================\n')

logging.info('all process done! exit now...')
