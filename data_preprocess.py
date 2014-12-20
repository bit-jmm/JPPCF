import sys
import os
import numpy as np
import csv
import datetime
import time

def get_user_like_and_doc_liked_list():
	data_path = './data/preprocessed_data/total_data/'
	rating_file = open(data_path + 'rating_data_file.dat.txt', 'r')

	user_writer = open(data_path + 'user_like_list.dat.txt', 'w+')
	doc_writer = open(data_path + 'doc_liked_list.dat.txt', 'w+')

	user_dict = {}
	doc_dict = {}

	for line in rating_file.readlines():
		splits = line.split()
		user_id = int(splits[0])
		doc_id = int(splits[1])

		if user_dict.has_key(user_id):
			user_dict[user_id].append(doc_id)
		else:
			user_dict[user_id] = [doc_id]
		if doc_dict.has_key(doc_id):
			doc_dict[doc_id].append(user_id)
		else:
			doc_dict[doc_id] = [user_id]
	sorted_user_list = sorted(user_dict.items(), lambda x,y: cmp(len(x[1]),len(y[1])), reverse = True)
	sorted_doc_list = sorted(doc_dict.items(), lambda x,y: cmp(len(x[1]),len(y[1])), reverse = True)
	
	for user in sorted_user_list:
		doc_str = ''
		for i in range(len(user[1])-1):
			doc_str += str(user[1][i]) + ' '
		doc_str += str(user[1][-1])
		user_writer.write(str(user[0]) + ' ' + str(len(user[1])) + ' ' + doc_str + '\n')

	for doc in sorted_doc_list:
		user_str = ''
		for i in range(len(doc[1])-1):
			user_str += str(doc[1][i]) + ' '
		user_str += str(doc[1][-1])
		doc_writer.write(str(doc[0]) + ' ' + str(len(doc[1])) + ' ' + user_str + '\n')

	rating_file.close()
	user_writer.close()
	doc_writer.close()


def filter_unactive_users_docs():
	data_path = './data/preprocessed_data/'
	rating_file = open(data_path + 'total_data/rating_data_file.dat.txt', 'r')
	user_like_list = open(data_path + 'total_data/user_like_list.dat.txt', 'r')
	doc_liked_list = open(data_path + 'total_data/doc_liked_list.dat.txt', 'r')
	threshold = 10

	print 'start filter ratings whose users and docs like list\'s len less than ', threshold 
	
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

	user_dict = {}
	doc_dict = {}
	time_step = 1

	for line in rating_file.readlines():
		splits = line.split()
		user_id = int(splits[0])
		doc_id = int(splits[1])
		time = splits[2]

		if(int(time) > time_step):
			filtered_user_like_list = open(result_path + '/user_like_list_at_time_step' + str(time_step) + '.dat.txt', 'w+')
			filtered_doc_liked_list = open(result_path + '/doc_liked_list_at_time_step' + str(time_step) + '.dat.txt', 'w+')
			sorted_user_list = sorted(user_dict.items(), lambda x,y: cmp(len(x[1]),len(y[1])), reverse = True)
			sorted_doc_list = sorted(doc_dict.items(), lambda x,y: cmp(len(x[1]),len(y[1])), reverse = True)
	
			for user in sorted_user_list:
				doc_str = ''
				for i in range(len(user[1])-1):
					doc_str += str(user[1][i]) + ' '
				doc_str += str(user[1][-1])
				filtered_user_like_list.write(str(user[0]) + ' ' + str(len(user[1])) + ' ' + doc_str + '\n')

			for doc in sorted_doc_list:
				user_str = ''
				for i in range(len(doc[1])-1):
					user_str += str(doc[1][i]) + ' '
				user_str += str(doc[1][-1])
				filtered_doc_liked_list.write(str(doc[0]) + ' ' + str(len(doc[1])) + ' ' + user_str + '\n')
			
			filtered_user_like_list.close()
			filtered_doc_liked_list.close()

			user_dict = {}
			doc_dict = {}
			time_step += 1


		if (user_id not in filtered_user_ids) or (doc_id not in filtered_doc_ids):
			continue
		fuser_id = filtered_user_ids.index(user_id) + 1
		fdoc_id = filtered_doc_ids.index(doc_id) + 1

		if user_dict.has_key(fuser_id):
			user_dict[fuser_id].append(fdoc_id)
		else:
			user_dict[fuser_id] = [fdoc_id]
		if doc_dict.has_key(fdoc_id):
			doc_dict[fdoc_id].append(fuser_id)
		else:
			doc_dict[fdoc_id] = [fuser_id]

		filter_rating_file.write(str(fuser_id) + ' ' + str(fdoc_id) + ' ' + time + '\n')

	filtered_user_like_list = open(result_path + '/user_like_list_at_time_step' + str(time_step) + '.dat.txt', 'w+')
	filtered_doc_liked_list = open(result_path + '/doc_liked_list_at_time_step' + str(time_step) + '.dat.txt', 'w+')
	sorted_user_list = sorted(user_dict.items(), lambda x,y: cmp(len(x[1]),len(y[1])), reverse = True)
	sorted_doc_list = sorted(doc_dict.items(), lambda x,y: cmp(len(x[1]),len(y[1])), reverse = True)

	for user in sorted_user_list:
		doc_str = ''
		for i in range(len(user[1])-1):
			doc_str += str(user[1][i]) + ' '
		doc_str += str(user[1][-1])
		filtered_user_like_list.write(str(user[0]) + ' ' + str(len(user[1])) + ' ' + doc_str + '\n')

	for doc in sorted_doc_list:
		user_str = ''
		for i in range(len(doc[1])-1):
			user_str += str(doc[1][i]) + ' '
		user_str += str(doc[1][-1])
		filtered_doc_liked_list.write(str(doc[0]) + ' ' + str(len(doc[1])) + ' ' + user_str + '\n')
	
	filtered_user_like_list.close()
	filtered_doc_liked_list.close()
	
	rating_file.close()
	user_like_list.close()
	doc_liked_list.close()
	filter_rating_file.close()

	print 'end.'
	
def cross_validate_on_dict(doc_liked_list, fold_num, time_result_path):
	folds_dict_list = []
	# the last dict store the like list for docs whose like list len lower than fold_num
	for i in range(fold_num+1):
		folds_dict_list.append({})
	for doc_id in doc_liked_list.keys():
		user_ids = doc_liked_list[doc_id]
		list_len = len(user_ids)
		if list_len < fold_num:
			folds_dict_list[fold_num][doc_id] = user_ids
			doc_liked_list.pop(doc_id)
		else:
			step_len = list_len / fold_num
			remainder = list_len % fold_num
			for i in range(fold_num):
				for j in range(i*step_len,(i+1)*step_len):
					if folds_dict_list[i].has_key(doc_id):
						folds_dict_list[i][doc_id].append(user_ids[j])
					else:
						folds_dict_list[i][doc_id] = [user_ids[j]]
				for j in range(list_len-remainder, list_len):
					if folds_dict_list[i].has_key(doc_id):
						folds_dict_list[i][doc_id].append(user_ids[j])
					else:
						folds_dict_list[i][doc_id] = [user_ids[j]]
	# end evenly split user-doc pair to folds
	# start generate train_data and test_data
	for i in range(fold_num):
		data_path = time_result_path + '/data_' + str(i)
		if (not os.path.isdir(data_path)):
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
					user_like_list_dict_in_test[user_id]= [doc_id]
				test_file.write(str(user_id) + ' ' + str(doc_id) + '\n')
		for fold_id in range(fold_num+1):
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



def generate_cross_validate_data(fold_num = 5):
	data_path = './data/preprocessed_data/'
	result_path = data_path + 'filtered_by_user_doc_like_list_len_10'
	rating_data = np.loadtxt(result_path + '/rating_file.dat.txt', dtype=int)
	start_time = rating_data[0][2]
	end_time = rating_data[-1][2]
	row, col = rating_data.shape
	doc_liked_dict = {}
	i = 0

	# generate data at every time step
	for time in range(start_time, end_time+1):
		time_path = result_path + '/time_step_' + str(time)
		if (not os.path.isdir(time_path)):
			os.mkdir(time_path)
		for j in range(i, row):
			user_id = rating_data[j][0]
			doc_id = rating_data[j][1]
			time_step = rating_data[j][2]
			if time_step != time:
				i = j
				cross_validate_on_dict(doc_liked_dict, fold_num, time_path)			
				doc_liked_dict = {}
				break
			if doc_liked_dict.has_key(doc_id):
				doc_liked_dict[doc_id].append(user_id)
			else:
				doc_liked_dict[doc_id] = [user_id]
			if j == row - 1:
				cross_validate_on_dict(doc_liked_dict, fold_num, time_path)
				i = row
				break

	
if __name__ == '__main__':    
	start = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	print 'process start at : ', start
	print '\n'

	#get_user_like_and_doc_liked_list()	
	filter_unactive_users_docs()
	generate_cross_validate_data(3)

	import time   
	end = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	print 'process end at : ', end

	print 'process take time : ', str(datetime.datetime.strptime(end,'%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,'%Y-%m-%d %H:%M:%S'))


