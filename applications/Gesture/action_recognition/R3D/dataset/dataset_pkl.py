import os 
import sys
# sys.path.append(os.getcwd()[0:-7])
# sys.path.append(os.path.join(os.getcwd()[0:-7], 'utils'))
import json
import pickle
import numpy as np
import pandas as pd
import random
import torch
import pdb
from torch.utils.data import Dataset, DataLoader,RandomSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import skimage.util as ski_util
from sklearn.utils import shuffle




# pdb.set_trace()

# self-defined modules


def load_video(annot_path, mode):
    # mode: train, val, test
    csv_file = os.path.join(annot_path, '{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    # annot_df = DownSample(raw_annot_df, 40)
    rgb_samples = []
    depth_samples = []
    labels = []
    # get task index in dataframe
    task_ind = []
    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df['frame'].iloc[frame_i] # convert string in dataframe to list
        rgb_samples.append(rgb_list)
        labels.append(annot_df['label'].iloc[frame_i])
    print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples, labels


class dataset_video(Dataset):
    def __init__(self, root_path, mode, spatial_transform=None, temporal_transform=None):
        self.root_path = root_path
        self.rgb_samples, self.labels = load_video(root_path, mode)
        rgb_test = Image.open(self.rgb_samples[0][0]).convert("RGB")
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        # print('{} {} samples have been loaded'.format(class_id, self.sample_num))

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_frames.append(rgb_cache)
        clip_frames = self.spatial_transform(clip_frames)
        return clip_frames.transpose(1,0), int(label)

    def __len__(self):
        return int(self.sample_num)





def load_video_test(annot_path, mode):
    # mode: train, val, test
    csv_file = os.path.join(annot_path, 'jester_{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    # annot_df = DownSample(raw_annot_df, 40)
    rgb_samples = []
    # get task index in dataframe
    task_ind = []
    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df['frame'].iloc[frame_i] # convert string in dataframe to list
        rgb_samples.append(rgb_list)
    print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples



class dataset_video_test(Dataset):
    def __init__(self, root_path, mode, spatial_transform=None, temporal_transform=None):
        self.root_path = root_path
        self.rgb_samples = load_video_test(root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        # print('{} {} samples have been loaded'.format(class_id, self.sample_num))

    def __getitem__(self, idx):
        clip_name = self.rgb_samples[idx]
        indices = [i for i in range(len(clip_name))]
        selected_indice = self.temporal_transform(indices)
        clip_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(clip_name[frame_name_i]).convert("RGB")
            clip_frames.append(rgb_cache)
        clip_frames = self.spatial_transform(clip_frames)
        return clip_name, clip_frames.transpose(1,0) 
        # return rgb, mask, (torch.tensor(label)-1).long()
    def __len__(self):
        return int(self.sample_num)



def get_label_dict_jester(label_path, filename):
    label = pd.read_csv(os.path.join(label_path, filename), header=None)
    label_list = []
    for i in range(len(label)):
        label_list.append(label.iloc[i].item())
    label_dict = {k: v for v,k in enumerate(label_list)}
    return label_dict


def get_label_dict_sthv2(label_path, filename):
    label_dict = dict()
    with open(os.path.join(label_path, 'something-something-v2-labels.json'), encoding='utf-8') as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i == int(idx)  # make sure the rank is right
        categories.append(cat)
        label_dict[cat] = i
    return label_dict




# from temporal_transforms import *
# from spatial_transforms import *
# from transforms import *

# from opts import parse_opts
# args = parse_opts()

# annot_path = '/home/zhengwei/workspace/something-try/SlowFastNetworks/annotations'

# args = parse_opts()
# scales = [args.initial_scale]
# for i in range(1, args.n_scales):
#     scales.append(scales[-1] * args.scale_step)

# # trans_train = Compose([
# #             # Scale([100,100]), # w * h: 176 * 100
# #             # SpatialElasticDisplacement(),
# #             MultiScaleRandomCrop(scales, [112, 112]),
# #             ToTensor(1)
# #             ])

# trans_train  = Compose([GroupMultiScaleCrop(112, [1, .875, .75, .66]),
#                         ToTorchFormatTensor(),
#                         Stack_3D(),
#                         GroupNormalize(mean=[.485, .456, .406], std=[.229, .224, .225])])

# temporal_transform_ = Compose([
#         TemporalRandomCrop(100)
#         # TemporalBeginCrop(100)
#         ])

# dataset_train = dataset_video(annot_path, 'train',
#                                     n_frames_per_clip=100, img_size=(112, 112),
#                                     reverse=False, transform=trans_train,
#                                     temporal_transform = temporal_transform_)
# rgb_name, rgbs = dataset_train.__getitem__(0)



# # dataloader_train = DataLoader(dataset_train, batch_size=32,
# #                                 shuffle=True, 
# #                                 num_workers=args.num_workers, pin_memory=True)
# # trainiter = iter(dataloader_train)
# # rgbs, masks, labels = trainiter.next()

# label_dict = get_label_dict_sthv2('/home/data2/zhengwei/sth-sth-v2', 'jester-v1-labels.csv')
# pdb.set_trace()