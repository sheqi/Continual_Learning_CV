"""
This script creates annotation file for training, validation and testing.
Run this file first before processing anything
"""

import pandas as pd
import os
import pdb
from tqdm import tqdm, trange
import numpy as np


# frame files stored path
frame_path = '/home/data2/zhengwei/jester/20bn-jester-v1' 
# label files stored path
label_path = '/home/data2/zhengwei/jester/'
save_path = '../jester_annotation'

if not os.path.exists(save_path):
    os.makedirs(save_path)

"""
jester-v1-train.csv
jester-v1-val.csv
jester-v1-test.csv
"""

def get_label_dict(label_path, filename):
    label = pd.read_csv(os.path.join(label_path, filename), header=None)
    label_list = []
    for i in range(len(label)):
        label_list.append(label.iloc[i].item())
    label_dict = {k: v for v,k in enumerate(label_list)}
    return label_dict


def construct_annot(frame_path, save_path, label_dict):
    files_input = ['jester-v1-validation.csv', 'jester-v1-train.csv', 'jester-v1-test.csv']
    files_output = ['val_videofolder.txt', 'train_videofolder.txt', 'test_videofolder.txt']
    files_output_pkl = ['val.pkl', 'train.pkl', 'test.pkl']


    # files_input = ['jester-v1-test.csv']
    # files_output = ['test_videofolder.txt']
    # files_output_pkl = ['test.pkl']

    for (filename_input, filename_output, filename_output_pkl) in zip(files_input, files_output, files_output_pkl):
        filename = os.path.join(label_path, filename_input)
        filelist = pd.read_csv(os.path.join(label_path, filename), header=None)
        annot_df = {'frame':[], 'label':[]}
        for i in trange(len(filelist)):
            if 'test' not in filename_input:
                folder_name, label_name = filelist.iloc[i].item().split(';')[0], filelist.iloc[i].item().split(';')[1]
            else:
                folder_name = str(filelist.iloc[i].item())
            frame_names = sorted(os.listdir(os.path.join(frame_path, folder_name)))
            video_names = []
            for frame_name in frame_names:
                video_names.append(os.path.join(frame_path, folder_name, frame_name))
            annot_df['frame'].append(video_names)
            if 'test' not in filename_input:
                label = label_dict[label_name]
            else:
                label = 0
            annot_df['label'].append(label)
        annot_df = pd.DataFrame(annot_df)
        annot_df.to_pickle(os.path.join(save_path, filename_output_pkl))


        annot_file = []
        with open(os.path.join(save_path, filename_output), 'w') as f:
            for i in trange(len(filelist)):
                if 'test' not in filename_input:
                    folder_name, label_name = filelist.iloc[i].item().split(';')[0], filelist.iloc[i].item().split(';')[1]
                else:
                     folder_name = str(filelist.iloc[i].item())
                if 'test' not in filename_input:
                    label = label_dict[label_name]
                else:
                    label = 0
                num_frame = len(sorted(os.listdir(os.path.join(frame_path, folder_name))))
                annot_file.append([folder_name, num_frame, label])
                f.write("{} {} {}\n".format(folder_name, num_frame, label))    



        # folder_name, num_frame, label
        



label_dict = get_label_dict(label_path, 'jester-v1-labels.csv')

# construct_annot(frame_path, 'jester-v1-train.csv', label_dict, 'train_videofolder.txt')
# construct_annot(frame_path, 'jester-v1-validation.csv', label_dict, 'val_videofolder.txt')
construct_annot(frame_path, save_path, label_dict)
