import sys
import os
import csv
import datetime
import time

days = 360
# set time interval (days)
time_interval = datetime.timedelta(days)
print 'set time_interval: ', time_interval

origin_data_path = './data/'
data_path = './data/preprocessed_data/'
matrices_data_path = data_path + 'data_divided_by_' + str(days) +'_days/'

print 'start prepare data...\n'

start = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print 'process start at : ', start
print '\n'

if not os.path.isdir(matrices_data_path):
    os.mkdir(matrices_data_path)

doc_word_datafile = open(origin_data_path + 'mult.dat.txt', 'r')
docs = doc_word_datafile.readlines()
doc_num = len(docs)

vocabulary_file = open(origin_data_path + 'vocab.dat.txt', 'r')
word_num = len(vocabulary_file.readlines())
print 'word_num: ', word_num

user_info_file = open(origin_data_path + 'user_name_id_map.csv', 'r')
user_num = len(user_info_file.readlines())-1
print 'user_num: ', user_num

doc_info_file = open(origin_data_path + 'doc_id_citeulike_id_map.csv', 'r')
doc_num = len(doc_info_file.readlines())-1
print 'doc_num: ', doc_num

rating_file = open(origin_data_path + 'user_doc_rating_time.csv', 'r')
rating_file_reader = csv.reader(rating_file)

doc_word_datafile.close()
vocabulary_file.close()
user_info_file.close()
doc_info_file.close()


ratings = {}
rating_list = []
doc_time_dict = {}
i = 0
for line in rating_file_reader:
    if i==0:
        i += 1
        continue
    time = datetime.datetime.strptime(line[3].split()[0], "%Y-%m-%d")
    max_time = time
    if i==1:
       min_time = time 
    if not doc_time_dict.has_key(line[1]):
        doc_time_dict[line[1]] = time
    ratings[line[0] + ',' + line[1]] = time
    rating_list.append((line[0],line[1],time))
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
    if processed_dict.has_key(key):
        continue
    else:
        processed_dict[key] = 1
    if i%10000 == 0:
            print ' i= ',i
    i += 1
    if l[2] > current_time:
        time_step += 1
        current_time += datetime.timedelta(days)
    writer.write(l[0] + ' ' + l[1] + ' ' + str(time_step) + '\n')
writer.close()

import time   
end = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print 'process end at : ', end

print 'process take time : ', str(datetime.datetime.strptime(end,'%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,'%Y-%m-%d %H:%M:%S'))
