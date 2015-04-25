import os
import time
from utility.data_preprocess import PrepareData
from model.ttarm import Ttarm
from model.trm import Trm


def prepare_data(time_step):
    start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print 'process start at : ', start
    print '\n'
    p = PrepareData(os.path.realpath( __file__ + '\..\data\preprocessed_data') +
        '\\', 10, 5, time_step)

    p.generate_rating_file_by_time_interval()
    p.get_user_like_and_doc_liked_list()
    p.filter_unactive_users_docs()
    p.generate_users_and_docs_dist()
    p.get_doc_id_citeulike_id_map()
    p.generate_cross_validate_data()

    end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print 'process end at : ', end

    print 'process take time : ', str(
        time.datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') -
        time.datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':

    time_step = 360
    # prepare_data(time_step)

    log_path = os.path.realpath(os.path.join(__file__, '../log'))
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    #model = Ttarm(k=20, lambd=10, eta=0.3, time_interval=time_step)
    model = Trm(k=20, lambd=10, time_interval=time_step)
    model.run()
