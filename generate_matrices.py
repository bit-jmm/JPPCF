import sys
import os
import csv
import datetime
import time

data_path = './data/preprocessed_data/'
matrices_data_path = data_path + 'matrices/'

print 'start prepare data...\n'

start = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print 'process start at : ', start
print '\n'

if not os.path.isdir(matrices_data_path):
    os.mkdir(matrices_data_path)
# set time interval (days)
time_interval = datetime.timedelta(180)
print 'set time_interval: ', time_interval

doc_word_datafile = open(data_path + 'mult.dat.txt', 'r')
docs = doc_word_datafile.readlines()
doc_num = len(docs)

vocabulary_file = open(data_path + 'vocab.dat.txt', 'r')
word_num = len(vocabulary_file.readlines())
print 'word_num: ', word_num

user_info_file = open(data_path + 'user_name_id_map.csv', 'r')
user_num = len(user_info_file.readlines())-1
print 'user_num: ', user_num

doc_info_file = open(data_path + 'doc_id_citeulike_id_map.csv', 'r')
doc_num = len(doc_info_file.readlines())-1
print 'doc_num: ', doc_num

rating_file = open(data_path + 'user_doc_rating_time.csv', 'r')
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
current_time = min_time + datetime.timedelta(180)
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
        current_time += datetime.timedelta(180)
    writer.write(l[0] + ' ' + l[1] + ' ' + str(time_step) + '\n')
writer.close()

##print '\nstart generate matrices data...\n'
##start_doc_id = 0
##start_rating_id = 0
##for time_step in range(0, time_step_num):
##    print 'start generate matrices at time step ', time_step
##    end_time = min_time + datetime.timedelta((time_step + 1)*180)
##
##    print '     end time: ', end_time
##    
##    timestep_path = matrices_data_path + '/timestep_' + str(time_step)
##    if not os.path.isdir(timestep_path):
##        os.mkdir(timestep_path)
##    doc_word_matrice = open(timestep_path + '/doc_word_matrice.dat.txt', 'w+')
##    
##    print '     generate doc word matrice...'
##    dnum = 0
##    for i in range(start_doc_id, doc_num):
##        start_doc_id = i
##        if i%100 == 0:
##            print '         i= ',i
##        if not doc_time_dict.has_key(str(i+1)):
##            continue
##        if doc_time_dict[str(i+1)] > end_time:
##            break
##        word_dict = {}
##        words = docs[i].split()
##        num = int(words[0])
##        for j in range(1, num+1):
##            split_string = words[j].split(':')
##            word_dict[int(split_string[0])] = int(split_string[1])
##        line = ''
##        for j in range(0, word_num):
##            if j < word_num-1:
##                if word_dict.has_key(j):
##                    line += str(word_dict[j]) + ' '
##                else:
##                    line += '0 '
##            else:
##                if word_dict.has_key(j):
##                    line += str(word_dict[j]) + '\n'
##                else:
##                    line += '0\n'
##        doc_word_matrice.write(line)
##        dnum += 1
##    print '         doc num at this time step: ', dnum
##    doc_word_matrice.close()
##
##    print '     end generate doc word matrice...\n'
##        
##    print '     start generate user doc matrice.'
##
##    user_doc_matrice = open(timestep_path + '/user_doc_matrice.dat.txt', 'w+')
##
##    for i in range(1, user_num+1):
##        line = ''
##        if i%3000 == 0:
##            print '         user_id= ',i
##        for j in range(1, doc_num+1):
##            key = str(i) + ',' +str(j)
##            if j < doc_num:
##                if(ratings.has_key(key) and ratings[key] <= end_time):
##                    line += '1 '
##                else:
##                    line += '0 '
##            else:
##                if(ratings.has_key(key) and ratings[key] <= end_time):
##                    line += '1\n'
##                else:
##                    line += '0\n'
##        user_doc_matrice.write(line)
##    user_doc_matrice.close()    
##    print '     end generate user doc matrice.\n'
    
import time   
end = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print 'process end at : ', end

print 'process take time : ', str(datetime.datetime.strptime(end,'%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,'%Y-%m-%d %H:%M:%S'))






