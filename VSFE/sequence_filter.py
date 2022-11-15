from glob import glob
import matplotlib.pyplot as plt
import os
import collections
import numpy as np
import pandas as pd
import pickle

rotate_filter_root = './vsfe_dataset/rotate_filtered_frames/'
video_name = glob(rotate_filter_root + '*/')

sequence_list = list()
for name in video_name:
    # frames = glob(name + '*')
    original_frame_length = len(glob(name.replace('rotate_filtered_frames', 'original_frames') + '*'))
    file_status = False
    sequence_single = list()
    for index in range(original_frame_length):
        frame_path = name + 'yaw_0.0_' + '{}_{}.jpg'.format(name.split('/')[-2], str(index).zfill(6))
        sequence_name = '/'.join(frame_path.split('/')[-2:])
        if file_status and os.path.isfile(frame_path):
            sequence_single.append(sequence_name)
        if not file_status and os.path.isfile(frame_path):
            file_status = True
            sequence_single.append(sequence_name)
        if file_status and not os.path.isfile(frame_path):
            file_status = False
            sequence_list.append(sequence_single)
            sequence_single = list()

dataset_list = list()
output_list = list()
sample_range = 16
for x in sequence_list:
    if len(x) >= 17:
        window_len = (len(x) - sample_range + 1)
        dataset_list.append(x)
        for y in range(window_len):
            window = x[y:y+sample_range]
            output_list.append(window)
#print(len(output_list))

with open('sequence_list_more_than_17.pkl', 'wb') as f:
    pickle.dump(dataset_list, f)

with open('sequence_list_range16.pkl', 'wb') as f:
    pickle.dump(output_list, f)


