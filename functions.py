import os
import scipy.integrate
import numpy as np
import pandas as pd
import ipaddress
from tqdm import tqdm
from feature_creation import feature_creation


heading = {0: ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Flow Duration',
                'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
                'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Std',
                'Bwd IAT Max', 'Bwd IAT Min'], # 20
           1: ['source_ip', 'source_port', 'destination_ip', 'destination_port', 'protocol', 'flow_duration',
                'flow_bytes', 'flow_packets', 'iat_p1', 'iat_p2', 'iat_p3', 'iat_p4', 'iat_p5', 'iat_p6', 'iat_p8',
                'forward_p1', 'forward_p2', 'forward_p3', 'forward_p4', 'forward_p5', 'forward_p6', 'forward_p8',
                'backward_p1', 'backward_p2', 'backward_p3', 'backward_p4', 'backward_p5', 'backward_p6',
               'backward_p8'], # 29
           2: ['source_ip', 'source_port', 'destination_ip', 'destination_port', 'protocol', 'flow_duration',
                'flow_bytes', 'flow_packets', 'iat_H', 'iat_A', 'iat_B', 'forward_iat_H', 'forward_iat_A',
               'forward_iat_B', 'backward_iat_H', 'backward_iat_A', 'backward_iat_B']} # 17


def walk_path(path):
    path = os.walk(path)
    paths = []

    for address, dirs, files in path:
        for file in files:
            file_path = address + '/' + file
            if '.csv' in file_path:
                paths.append(file_path)
    return paths


def get_path(path):
    return (os.path.join(path, f)
            for f in os.listdir(path)
            if 'csv' in f)


def value(array, mean):
    val = []
    range_array = max(array) - min(array)
    for index in range(len(array)):
        area = (array[index] - mean) / range_array
        val.append(area)
    return val


def trap_sum(array):
    list_trap_sum = []
    for index in range(1, len(array) + 1):
        area = scipy.integrate.trapz(array[0:index])
        list_trap_sum.append(area)
    return list_trap_sum


def create_flow(table):
    # Create forward packet tables
    table_forward = table.loc[(table['Source IP'] == table.iloc[0]['Source IP'])]
    table_forward = table_forward.loc[(table_forward['Source Port'] == table.iloc[0]['Source Port'])]
    table_forward = table_forward.loc[(table_forward['Destination IP'] == table.iloc[0]['Destination IP'])]
    table_forward = table_forward.loc[(table_forward['Destination Port'] == table.iloc[0]['Destination Port'])]
    table_forward.reset_index(drop=True, inplace=True)

    # Create backward packet tables
    table_backward = table.loc[(table['Destination IP'] == table.iloc[0]['Source IP'])]
    table_backward = table_backward.loc[(table_backward['Destination Port'] == table.iloc[0]['Source Port'])]
    table_backward = table_backward.loc[(table_backward['Source IP'] == table.iloc[0]['Destination IP'])]
    table_backward = table_backward.loc[(table_backward['Source Port'] == table.iloc[0]['Destination Port'])]
    table_backward.reset_index(drop=True, inplace=True)

    if ipaddress.ip_address(table_forward.iloc[0]['Destination IP']).is_private:
        table_forward, table_backward = table_backward, table_forward

    # Clear table
    table = pd.concat([table, table_forward, table_backward]).drop_duplicates(keep=False)
    table.reset_index(drop=True, inplace=True)
    table_flow = pd.concat([table_forward, table_backward]).sort_values(by=['index']).reset_index(drop=True)
    return table, table_forward, table_backward, table_flow


def temp_flow(tmp, tmp_flow_table):
    tmp = np.concatenate((tmp_flow_table, tmp))
    unq, ind, count = np.unique(tmp[:, 0], return_index=True, return_counts=True)
    count = count[np.argsort(ind)]
    index = np.resize(np.argwhere(count > 1), len(np.argwhere(count > 1)))
    tmp = tmp[index]
    times = np.subtract(tmp[1:, 7], tmp[:-1, 7])
    times = times * 1e+6
    return times


def flow_statistics(times, flow, time, start_time, frames_lenght, table_forward, table_backward, tmp_flow_table,
                    table_flow, statistic, idx, features_num):
    times = times * 1e+6

    # Statistics of flow
    flow_duration = (time - start_time) * 1e+6
    flow_bytes = np.sum(frames_lenght) / (np.sum(times) * 1e-6)
    flow_packets = len(frames_lenght) / (np.sum(times) * 1e-6)
    tmp_statistic = np.expand_dims(np.append([flow], [flow_duration, flow_bytes, flow_packets]), axis=0)

    # Create iat features
    features = feature_creation(times, features_num)
    tmp_statistic = np.expand_dims(np.append([tmp_statistic], [features]), axis=0)

    # Create temp forward flow
    times = temp_flow(table_forward, tmp_flow_table)
    features = feature_creation(times, features_num)
    tmp_statistic = np.expand_dims(np.append([tmp_statistic], [features]), axis=0)

    # Create temp backward flow
    times = temp_flow(table_backward, tmp_flow_table)
    features = feature_creation(times, features_num)
    tmp_statistic = np.expand_dims(np.append([tmp_statistic], [features]), axis=0)

    start_time = table_flow[idx, 7]
    time = table_flow[idx, 7]
    times = np.array([])
    frames_lenght = table_flow[idx, 6]

    tmp_flow_table = np.empty([0, 8])

    statistic = np.append(statistic, tmp_statistic, axis=0)
    return statistic, start_time, time, times, frames_lenght, tmp_flow_table


def iat_statistics(file, timeout, label, features_num):
    table = pd.read_csv(file)
    table_statistics = np.empty([0, len(heading[features_num])])
    while len(table) > 0:
        table, table_forward, table_backward, table_flow = create_flow(table)
        if len(table_forward) > 2 and len(table_backward) > 2:
            start_time = 0
            times = np.array([])
            statistic = np.empty([0, len(heading[features_num])])
            flow = np.array([table_forward['Source IP'].iloc[0], table_forward['Source Port'].iloc[0],
                             table_backward['Source IP'].iloc[0], table_backward['Source Port'].iloc[0],
                             table_forward['Protocol'].iloc[0]])
            table_flow = table_flow.to_numpy()
            table_forward = table_forward.to_numpy()
            table_backward = table_backward.to_numpy()

            for idx in tqdm(range(len(table_flow))):
                if start_time == 0:
                    start_time = table_flow[idx, 7]
                    time = table_flow[idx, 7]
                    frames_lenght = np.array(table_flow[idx, 6])
                    tmp_flow_table = np.expand_dims(table_flow[idx], axis=0)
                elif table_flow[idx, 7] - start_time > timeout:
                    if len(times) > 1:
                        statistic, start_time, time, times, frames_lenght, tmp_flow_table = flow_statistics(times,
                                                                                                            flow, time,
                                                                                                            start_time,
                                                                                                            frames_lenght,
                                                                                                            table_forward,
                                                                                                            table_backward,
                                                                                                            tmp_flow_table,
                                                                                                            table_flow,
                                                                                                            statistic,
                                                                                                            idx, features_num)

                elif idx == len(table_flow) - 1:
                    if len(times) > 1:
                        statistic, start_time, time, times, frames_lenght, tmp_flow_table = flow_statistics(times, flow,
                                                                                                            time,
                                                                                                            start_time,
                                                                                                            frames_lenght,
                                                                                                            table_forward,
                                                                                                            table_backward,
                                                                                                            tmp_flow_table,
                                                                                                            table_flow,
                                                                                                            statistic,
                                                                                                            idx, features_num)

                else:
                    times = np.append(times, table_flow[idx, 7] - time)
                    time = table_flow[idx, 7]
                    frames_lenght = np.append(frames_lenght, table_flow[idx, 6])
                    tmp_flow_table = np.append(tmp_flow_table, np.expand_dims(table_flow[idx], axis=0), axis=0)

            table_statistics = np.append(table_statistics, statistic, axis=0)

    table_statistics = pd.DataFrame(table_statistics, columns=heading[features_num])
    table_statistics['label'] = label
    table_statistics.to_csv(
        path_or_buf='tmp/' + os.path.splitext(os.path.splitext(os.path.split(file)[1])[0])[0] + '_features_num_' +
                    str(features_num)+ '.csv', index=False)

    return 0
