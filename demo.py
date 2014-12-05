import os
import numpy as np
import util
from JPPCF import *
data_path = './data/preprocessed_data/filtered_by_user_doc_like_list_len_5/'
R = np.loadtxt(data_path + 'rating_file.dat.txt', int)
#row_num = 50000
#R = R[range(row_num)][:]
user_num = R[:, 0].max()
doc_num = R[:, 1].max()
time_step_num = R[:, 2].max()

print 'user num: ', user_num, '\n'
print 'doc num: ', doc_num, '\n'
print 'time step num: ', time_step_num, '\n'

# We fix the num of latent feature
#k = 200

# flag variable
JPPflag = True


regl1nmf = 0.0005

regl1jpp = 0.05

epsilon = 0.1

maxiter = 100

#recall_num = 100

fold_num = 5

Rall = util.generate_matrice_between_time(R, user_num,
										  doc_num, 1, time_step_num)
#Rall = np.ones((100, 50))

print Rall.shape, type(Rall)

result_dir = './result/recall_by_cross_validate_fold_' + str(fold_num) + '_3models_lambda_0.5'
if not os.path.isdir(result_dir):
	os.mkdir(result_dir)

for k in [100]:

	for lambd in [0.5]:
		MR = []
		MRbaseline = []


		# the start time period used for init of W(1) and H(1), using normal NMF
		for start in [1]:

			Rt = util.generate_matrice_between_time(R, user_num,
													doc_num, start, start)

			#Rt = np.ones((100, 50))
			print 'non zero cell num: ', len(np.nonzero(Rt)[0])
			(P, Q) = util.nmf(Rt, k, maxiter, regl1nmf, epsilon)
			print P.shape, Q.shape

			# number of period we consider
			finT = time_step_num

			#for all the consecutive periods
			for current_time_step in range(start+1, finT + 1):

				print '\n=========================\n'
				print 'time_step number %i:\n', current_time_step
				print '----------------\n'

				#Rtall = util.generate_matrice_between_time(R, user_num,
														#doc_num,
														#current_time_step,
														#current_time_step)
				Po = P
				
				trecall_dict = {}
				frecall_dict = {}
				jrecall_dict = {}
				#for fold_id in range(fold_num):
				for fold_id in [0]:
					train_data_path = data_path + 'time_step_' + str(current_time_step) + '/data_' + str(fold_id) + '/train.dat.txt'
					Rt = util.generate_matrice_for_file(train_data_path, user_num, doc_num)
					print 'non zero cell num: ', len(np.nonzero(Rt)[0])
					#Rt = np.ones((100, 50))

					#P = np.ones((100,6))

					print 'computing JPP decomposition...'
					P, Q, S = JPPCF(Rt, Po, Po.shape[1], lambd, regl1jpp,  epsilon, maxiter, True)
					PredictR = P.dot(Q)
					NormPR = PredictR / PredictR.max()
					#NormPR = util.norm_by_threshold(NormPR, 0.5)

					
					print '[ok]\ncomputing t-model NMF decomposition...'
					Pbaseline, Qbaseline = util.nmf(Rt, k, maxiter, regl1nmf, epsilon)

					PredictRbaseline = Pbaseline.dot(Qbaseline)
					NormPRbaseline = PredictRbaseline / PredictRbaseline.max()

					print '[ok]\ncomputing fix_model NMF decomposition...'
					Rt = util.generate_matrice_between_time(R, user_num,
															doc_num,
															start + 1,
															current_time_step-1,
															train_data_path)
					#Rt = np.ones((100, 50))
					print 'non zero cell num: ', len(np.nonzero(Rt)[0])
					Pbaseline2, Qbaseline2 = util.nmf(Rt, k, maxiter, regl1nmf, epsilon)

					PredictRbaseline2 = Pbaseline2.dot(Qbaseline2)
					NormPRbaseline2 = PredictRbaseline2 / PredictRbaseline2.max()

					print '[ok]\n'
					#NormPRbaseline = util.norm_by_threshold(NormPRbaseline, 0.5)

					# RMSE = util.performanceRMSE(NormPRbaseline, Rall)
					#nmf_recall = util.performance_recall(NormPRbaseline, data_path, current_time_step, recall_num)
					#MRbaseline.append(recall)
					#print 't-model NMF RMSE: ', RMSE, '\n'
					#print 't-model NMF recall at ' + str(recall_num) + ' : ', recall, '\n'

					#if JPPflag:
						# RMSE = util.performanceRMSE(NormPR, Rall)
						#jppcf_recall = util.performance_recall(NormPR, data_path, current_time_step, recall_num)
						#MR.append(recall)
						#print 'JPPCF RMSE: ', RMSE, '\n'
						#print 'JPPCF recall at ' + str(recall_num) + ' : ', recall, '\n'

					print '\t fold_id:', fold_id, '\n'
					for recall_num in [10,50,100,300,500,1000]:
						print '\trecall at ',recall_num, ':'
						#tnmf_recall = util.performance_recall(NormPRbaseline, data_path, current_time_step, recall_num)
						#fnmf_recall = util.performance_recall(NormPRbaseline2, data_path, current_time_step, recall_num)
						#jppcf_recall = util.performance_recall(NormPR, data_path, current_time_step, recall_num)
						current_data_path = data_path + 'time_step_' + str(current_time_step) + '/data_' + str(fold_id)
						tnmf_recall = util.performance_cross_validate_recall(NormPRbaseline, current_data_path, recall_num)
						fnmf_recall = util.performance_cross_validate_recall(NormPRbaseline2, current_data_path, recall_num)
						jppcf_recall = util.performance_cross_validate_recall(NormPR, current_data_path, recall_num)
						
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
						print '\t\tt-model NMF :  ', tnmf_recall, '\n'
						print '\t\tf-model NMF :  ', fnmf_recall, '\n'
						print '\t\tJPPCF :  ', jppcf_recall, '\n'

				print 'current_time_step: ', current_time_step, '\n'
				for recall_num in [10,50,100,300,500,1000]:
					print '\tcross validate recall at ',recall_num, ':'
					avg_tnmf_recall = util.avg_of_list(trecall_dict[recall_num])
					avg_fnmf_recall = util.avg_of_list(frecall_dict[recall_num])
					avg_jppcf_recall = util.avg_of_list(jrecall_dict[recall_num])
					print '\t\tavg t-model NMF :  ', avg_tnmf_recall, '\n'
					print '\t\tavg f-model NMF :  ', avg_fnmf_recall, '\n'
					print '\t\tavg JPPCF :  ', avg_jppcf_recall, '\n'

					result_file = open(result_dir + '/k_' + str(k) +  '_lambda_' + str(lambd) + '_recall_at_' + str(recall_num)  + '.txt', 'a')
					result_file.write(str(avg_tnmf_recall) + '\t' + str(avg_fnmf_recall) + '\t' + str(avg_jppcf_recall) + '\n')
					result_file.close()

				print '=========================\n'
				#print 'computing JPPCF deposition for PO.... '
				#Rtall = util.generate_matrice_between_time(R, user_num,
														#doc_num,
														#current_time_step,
														#current_time_step)
				##Rtall = np.ones((100, 50))
				#P, Q, S = JPPCF(Rtall, Po, Po.shape[1], lambd, regl1jpp,  epsilon, maxiter, True)

			#mmr = np.mean(MR)
			##print 'JPPCF Avg RMSE: ', mmr, '\n'
			#print 'JPPCF Avg recall at ' + str(recall_num) + ' : ', mmr, '\n'
			#mmr = np.mean(MRbaseline)
			##print 't-model Avg NMF RMSE: ', mmr, '\n'
			#print 't-model NMF Avg recall at ' + str(recall_num) + ' : ', mmr, '\n'

			#num = len(MR)

			#mr = np.zeros((2, num))

			#for i in range(num):
				#mr[0][i] = MRbaseline[i]
				#mr[1][i] = MR[i]
			#np.savetxt('./result/k_' + str(k) +  '_lambda_' + str(lambd) + '_recall_at_' + str(recall_num)  + '.txt', mr)
