import os
import sys
# sys.path.append(os.getcwd()[0:-7])
# sys.path.append(os.path.join(os.getcwd()[0:-7], 'utils'))
import pickle
import numpy as np
import pandas as pd
import random
import torch
import pdb
from torch.utils.data import Dataset, DataLoader, RandomSampler
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
    csv_file = os.path.join(annot_path, 'EgoGesture_{}.pkl'.format(mode))
    annot_df = pd.read_pickle(csv_file)
    # annot_df = DownSample(raw_annot_df, 40)
    rgb_samples = []
    depth_samples = []
    labels = []
    # get task index in dataframe
    task_ind = []
    for frame_i in range(annot_df.shape[0]):
        rgb_list = annot_df['rgb'].iloc[frame_i]  # convert string in dataframe to list
        rgb_samples.append(rgb_list)
        depth_list = annot_df['depth'].iloc[frame_i]  # convert string in dataframe to list
        depth_samples.append(depth_list)
        labels.append(annot_df['label'].iloc[frame_i])
        # data augmentation by reversing the sequence of the video
    print('{}: {} videos have been loaded'.format(mode, len(rgb_samples)))
    return rgb_samples, depth_samples, labels


class dataset_video(Dataset):
    def __init__(self, root_path, mode, spatial_transform, temporal_transform):
        self.root_path = root_path
        self.rgb_samples, self.depth_samples, self.labels = load_video(root_path, mode)
        rgb_test = Image.open(self.rgb_samples[0][0]).convert("RGB")
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        # print('{} {} samples have been loaded'.format(class_id, self.sample_num))

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        depth_name = self.depth_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_rgb_frames = []
        clip_depth_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_rgb_frames.append(rgb_cache)
            depth_cache = Image.open(depth_name[frame_name_i]).convert("L")
            clip_depth_frames.append(depth_cache)
        clip_rgb_frames = self.spatial_transform(clip_rgb_frames)
        clip_depth_frames = self.spatial_transform(clip_depth_frames)
        return clip_rgb_frames, clip_depth_frames, int(label)
        # return rgb, mask, (torch.tensor(label)-1).long()

    def __len__(self):
        return int(self.sample_num)

# from temporal_transforms import *
# from spatial_transforms import *
# from transforms import *


# annot_path = '../annotation'


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
#         TemporalRandomCrop(16)
#         # TemporalBeginCrop(100)
#         ])

# dataset_train = dataset_video(annot_path, 'train', spatial_transform=trans_train, temporal_transform = temporal_transform_)
# rgb, depth, label = dataset_train.__getitem__(0)


# # dataloader_train = DataLoader(dataset_train, batch_size=32,
# #                                 shuffle=True, 
# #                                 num_workers=args.num_workers, pin_memory=True)
# # trainiter = iter(dataloader_train)
# # rgbs, masks, labels = trainiter.next()

# pdb.set_trace()
