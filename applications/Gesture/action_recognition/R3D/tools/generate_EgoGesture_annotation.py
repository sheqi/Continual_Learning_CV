"""
This script creates annotation file for training, validation and testing.
Run this file first before processing anything
"""

import pandas as pd
import os
import pdb
from tqdm import tqdm, trange
import numpy as np

# def construct_annot(frame_path, filename, label_dict, save_file_name):
#     filename = os.path.join(label_path, filename)
#     filelist = pd.read_csv(os.path.join(label_path, filename), header=None)
#     annot_file = []
#     with open(os.path.join(label_path, save_file_name), 'w') as f:
#         for i in trange(len(filelist)):
#             folder_name, label_name = filelist.iloc[i].item().split(';')[0], filelist.iloc[i].item().split(';')[1]
#             label = label_dict[label_name]
#             num_frame = len(sorted(os.listdir(os.path.join(frame_path, folder_name))))
#             annot_file.append([folder_name, num_frame, label])
#             f.write("{} {} {}\n".format(folder_name, num_frame, label))    



# frame files stored path
frame_path = '/home/data/egogesture/' 
# label files stored path
label_path = '/home/data/egogesture/labels-final-revised1'
save_dir = '../EgoGesture_annotation'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)




training_id = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50]
validating_id = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
testing_id = [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]


# for training
def create_annotation(save_dir, save_filename, ids):
    annot_dict = {k: [] for k in ['rgb', 'depth', 'label']}
    for sub_i in tqdm(ids):
        frame_path_sub = os.path.join(frame_path, 'Subject{:02}'.format(sub_i))
        label_path_sub = os.path.join(label_path, 'Subject{:02}'.format(sub_i))
        assert len([name for name in os.listdir(label_path_sub) if name != '.DS_Store']) == len([name for name in os.listdir(frame_path_sub)])
        for scene_i in range(1, len([name for name in os.listdir(frame_path_sub)])+1):
            rgb_path = os.path.join(frame_path_sub, 'Scene{:01}'.format(scene_i), 'Color')
            depth_path = os.path.join(frame_path_sub, 'Scene{:01}'.format(scene_i), 'Depth')
            label_path_iter = os.path.join(label_path_sub, 'Scene{:01}'.format(scene_i))
            assert len([name for name in os.listdir(label_path_iter) if 'csv'==name[-3::]]) == len([name for name in os.listdir(rgb_path)])
            assert len([name for name in os.listdir(label_path_iter) if 'csv'==name[-3::]]) == len([name for name in os.listdir(depth_path)])
            for group_i in range(1, len([name for name in os.listdir(rgb_path)])+1):
                rgb_path_group = os.path.join(rgb_path, 'rgb{:01}'.format(group_i))
                depth_path_group = os.path.join(depth_path, 'depth{:01}'.format(group_i))
                if os.path.isfile(os.path.join(label_path_iter, 'Group{:01}.csv'.format(group_i))):
                    label_path_group = os.path.join(label_path_iter, 'Group{:01}.csv'.format(group_i))
                else:
                    label_path_group = os.path.join(label_path_iter, 'group{:01}.csv'.format(group_i))
                # read the annotation files in the label path
                data_note = pd.read_csv(label_path_group, names = ['class', 'start', 'end'])
                data_note = data_note[np.isnan(data_note['start']) == False]
                for data_i in range(data_note.values.shape[0]):
                    label = data_note.values[data_i,0]
                    rgb = []
                    depth = []
                    for img_ind in range(int(data_note.values[data_i,1]), int(data_note.values[data_i,2]-1)):
                        rgb.append(os.path.join(rgb_path_group, '{:06}.jpg'.format(img_ind)))
                        depth.append(os.path.join(depth_path_group, '{:06}.jpg'.format(img_ind)))
                    annot_dict['rgb'].append(rgb) 
                    annot_dict['depth'].append(depth)
                    annot_dict['label'].append(label-1)
    annot_df = pd.DataFrame(annot_dict)
    save_file = os.path.join(save_dir, save_filename)
    annot_df.to_pickle(save_file)

create_annotation(save_dir, 'train.pkl', training_id)
create_annotation(save_dir, 'val.pkl', validating_id)
create_annotation(save_dir, 'test.pkl', testing_id)




def construct_annot(file_path, mode):       
    csv_file = os.path.join(file_path, '{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    # for i in trange(len(annot_df)):
    #     rgb_folders.append(annot_df.iloc[i].rgb[0][0:-11])
    #     depth_folders.append(annot_df.iloc[i].depth[0][0:-11])
    #     labels.append(annot_df.iloc[i].label)

    with open(os.path.join(file_path, '{}_videofolder.txt'.format(mode)), 'w') as f:
        for i in trange(len(annot_df)):
            f.write("{} {} {}\n".format(annot_df.iloc[i].rgb[0][0:-11], len(annot_df.iloc[i].rgb), annot_df.iloc[i].label-1)) 

    # with open(os.path.join('./', '{}_EgoGesture_depth.txt'.format(mode)), 'w') as f:
    #     for i in trange(len(annot_df)):
    #         f.write("{} {} {}\n".format(annot_df.iloc[i].depth[0][0:-11], len(annot_df.iloc[i].depth), annot_df.iloc[i].label-1)) 

categories = [i for i in range(0,83)]
with open(os.path.join(save_dir, 'category.txt'), 'w') as f:
    for category in categories:
        f.write('{}\n'.format(category))
construct_annot(save_dir, 'train')
construct_annot(save_dir, 'val')
construct_annot(save_dir, 'test')
