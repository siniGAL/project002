from functions import iat_statistics
from functions import walk_path
import time as timer
import shutil
import os
import pandas as pd
from multiprocessing import Pool
import itertools


def multi_function(file, timeout, features_num):
    if 'audio' in file.lower():
        label = 'Audio'
    elif 'browser' in file.lower():
        label = 'Browsing'
    elif 'chat' in file.lower():
        label = 'CHAT'
    elif 'file-transfer' in file.lower():
        label = 'FT'
    elif 'mail' in file.lower():
        label = 'Mail'
    elif 'p2p' or 'tor' in file.lower():
        label = 'P2P'
    elif 'video' in file.lower():
        label = 'Video'
    elif 'voip' in file.lower():
        label = 'VOIP'

    iat_statistics(file, timeout, label, features_num)

if __name__  == '__main__':
    timeout = 120
    features_num = 0    # [0-2]

    source_path = '/Users/macbookpro/Documents/all_numpy/csv_of_pcap/tor'
    files = walk_path(source_path)
    shutil.rmtree('tmp', ignore_errors=True, onerror=None)
    os.mkdir('tmp', dir_fd=None)

    start_timer = timer.time()
    pool = Pool(8)
    pool.starmap(multi_function, zip(files, list(itertools.repeat(timeout, len(files))),
                                     list(itertools.repeat(features_num, len(files)))))
    pool.close()
    pool.join()
    files = walk_path('tmp')
    data = pd.DataFrame()

    for file in files:
        data_tmp = pd.read_csv(file, sep=",")
        data = pd.concat([data, data_tmp])
    data.to_csv('statistics_' + str(timeout) + 's' + '_features_num_' + str(features_num) + '_scenario_a' + '.csv',
                index=False)

    shutil.rmtree('tmp', ignore_errors=False, onerror=None)
    print('Time to complete: %s' % ((timer.time() - start_timer) / 60))
