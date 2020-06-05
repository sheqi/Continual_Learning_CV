import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm, trange
import shutil
from random import randint
import argparse
import glob
import pdb
import random
import math
import time
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchsummary import summary

from models import resnext
from model import generate_model
import utils
# from dataset.spatial_transforms import *
from dataset.transforms import *
from dataset.temporal_transforms import *
from dataset import dataset_EgoGesture

import warnings
import os

# os.environ['CUDA_VISIBLE_DEVICES']='3'
warnings.filterwarnings("ignore")


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0,1,2,3')

    # args for dataloader
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_val', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--w', type=int, default=112)
    parser.add_argument('--h', type=int, default=112)
    parser.add_argument('--clip_len', type=int, default=32)

    # args for generating the model
    parser.add_argument('--model', type=str, default='resnext')
    parser.add_argument('--arch', type=str, default='resnext-101')
    parser.add_argument('--model_depth', type=int, default=101)
    parser.add_argument('--sample_size', type=int, default=112)
    parser.add_argument('--resnet_shortcut', type=str, default='B')
    parser.add_argument('--resnext_cardinality', type=int, default=32)
    # parser.add_argument('--sample_duration', type=int, default=32)
    parser.add_argument('--pretrain_path', type=str,
                        default='/home/data2/zhengwei/R3D_pretrained_kinetics/resnext-101-kinetics.pth',
                        # default='models/pretrained_models/resnext-101-kinetics.pth',
                        # default = 'models/pretrained_models/resnet-50-kinetics.pth',
                        help='Pretrained path for training model, DO NOT use for testing. Testing trained path is \
    defined in the main script')
    parser.add_argument('--modality', type=str, default='Depth',
                        help='Modality of input data. RGB, Depth, RGB-D and fusion. Fusion \
                            is only used when testing the two steam model')
    parser.add_argument('--n_classes', type=int, default=83)
    parser.add_argument('--n_finetune_classes', type=int, default=83)
    parser.add_argument('--ft_begin_index', type=int, default=0,
                        help='How many parameters need to be fine tuned')
    parser.add_argument('--no_cuda', type=bool, default=False)

    # args for preprocessing
    parser.add_argument('--initial_scale', type=float, default=1,
                        help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int,
                        help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float,
                        help='Scale step for multiscale cropping')

    # args for training
    parser.add_argument('--lr_steps', type=list, default=[10, 20],
                        help='lr steps for decreasing learning rate')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    args = parser.parse_args()
    return args


args = parse_opts()

annot_dir = './annotation'
save_dir = '{}-{}f-{}'.format(args.arch, args.clip_len, args.modality)

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
device = 'cuda:0'
if isinstance(args.cuda_id, list):
    device_ids = [i for i in eval(args.cuda_id)]
else:
    device_ids = [eval(args.cuda_id)]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def forward(model, data):
    rgbs, depths, labels = data
    if args.modality == 'RGB':
        inputs = rgbs.to(device, non_blocking=True).float()
    elif args.modality == 'Depth':
        inputs = depths.to(device, non_blocking=True).float()
    elif args.modality == 'RGB-D':
        inputs = torch.cat((rgbs, depths), 1).to(device, non_blocking=True).float()
    probs, logits = model(inputs)
    labels = labels.to(device, non_blocking=True).long()
    return probs, logits, labels


def model_test(model, save_dir, filename, dataloader, num_class):
    model.module.fc = nn.Linear(model.module.fc.in_features, num_class)
    model.module.fc.to(device)
    checkpoint = utils.load_checkpoint(save_dir, filename)
    # model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print('Evaluating for model {}........'.format(filename))
    acc = utils.AverageMeter()
    for data in tqdm(dataloader):
        probs, logits, labels = forward(model, data)
        acc.update(utils.calculate_accuracy(probs, labels))
    print('val_acc:{:.3f}'.format(acc.avg))


def model_train(model, save_dir, dataloader_train, dataloader_val):
    model.train()
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()

    # determine optimizer
    fc_lr_layers = list(map(id, model.module.fc.parameters()))
    pretrained_lr_layers = [p for p in model.parameters()
                            if id(p) not in fc_lr_layers and p.requires_grad == True]
    # pretrained_lr_layers = filter(lambda p: 
    #                               id(p) not in fc_lr_layers, model.parameters())
    # optimizer = torch.optim.SGD([
    #     {"params": model.module.fc.parameters()},
    #     {"params": pretrained_lr_layers, "lr": 1e-4, 'weight_decay':1e-3}
    # ], lr=1e-3, momentum=0.9, weight_decay=1e-3)    

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)

    train_logger = utils.Logger(os.path.join(save_dir, '{}-{}-{}.log'.format(args.arch, args.clip_len, args.modality)),
                                ['step', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
                                 'lr_feature', 'lr_fc'])
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()

    step = 0
    for epoch in trange(num_epochs):  # loop over the dataset multiple times
        train_loss.reset()
        train_acc.reset()
        for data in dataloader_train:
            probs, outputs, labels = forward(model, data)
            optimizer.zero_grad()
            loss_ = criterion(outputs, labels)
            loss_.backward()
            optimizer.step()
            train_loss.update(loss_.item())
            train_acc.update(utils.calculate_accuracy(probs, labels))
            if step % 100 == 0:
                val_loss.reset()
                val_acc.reset()
                model.eval()
                for data_val in dataloader_val:
                    probs_val, outputs_val, labels_val = forward(model, data_val)
                    val_loss_ = criterion(outputs_val, labels_val)
                    val_loss.update(val_loss_.item())
                    val_acc.update(utils.calculate_accuracy(probs_val, labels_val))
                model.train()
                print('epoch{}/{} train_acc:{:.3f} train_loss:{:.3f} val_acc:{:.3f} val_loss:{:.3f}'.format(
                    epoch + 1, num_epochs,
                    train_acc.val, train_loss.val,
                    val_acc.avg, val_loss.avg
                ))
                train_logger.log({
                    'step': step,
                    'train_loss': train_loss.val,
                    'train_acc': train_acc.val,
                    'val_loss': val_loss.avg,
                    'val_acc': val_acc.avg,
                    # 'lr_feature': optimizer.param_groups[1]['lr'],
                    'lr_feature': 0,
                    'lr_fc': optimizer.param_groups[0]['lr']
                })
            step += 1
        utils.save_checkpoint(model, optimizer, step, save_dir,
                              '{}-{}-{}.pth'.format(args.arch, args.clip_len, args.modality))
        # scheduler.step()
        utils.adjust_learning_rate(args.learning_rate, optimizer, epoch, args.lr_steps)


if __name__ == '__main__':
    # keep shuffling be constant every time
    seed = 1
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    mean = [.485, .456, .406]
    std = [.229, .224, .225]

    scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        scales.append(scales[-1] * args.scale_step)

    temporal_transform_train = transforms.Compose([
        TemporalRandomCrop(args.clip_len)
    ])

    temporal_transform_test = transforms.Compose([
        TemporalCenterCrop(args.clip_len)
    ])

    trans_train = transforms.Compose([
        GroupScale([140, 140]),  # 112/0.8
        GroupMultiScaleCrop([args.w, args.h], scales),
        # GroupMultiScaleRotate(20),
        ToTorchFormatTensor(),
        Stack_3D(),
        GroupNormalize(mean=mean, std=std)])

    trans_test = transforms.Compose([
        GroupScale([args.w, args.h]),
        ToTorchFormatTensor(),
        Stack_3D(),
        GroupNormalize(mean=mean, std=std)])

    # load dataset
    if args.is_train:
        print('Loading training data.....')
        class_id1 = [i for i in range(1, 41)]
        dataset_train = dataset_EgoGesture.dataset_video(annot_dir, 'train',
                                                         spatial_transform=trans_train,
                                                         temporal_transform=temporal_transform_train)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)

        print('\n')
        print('Loading validating data.....')
        dataset_val = dataset_EgoGesture.dataset_video(annot_dir, 'val',
                                                       spatial_transform=trans_test,
                                                       temporal_transform=temporal_transform_test)
        dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size_val,
                                    num_workers=args.num_workers, pin_memory=True)



    else:
        print('Loading testing data.....')
        class_id1 = [i for i in range(1, 41)]
        dataset_test = dataset_EgoGesture.dataset_video(annot_dir, 'test',
                                                        spatial_transform=trans_test,
                                                        temporal_transform=temporal_transform_test)
        dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size_val,
                                     num_workers=args.num_workers, pin_memory=True)

    model, parameters = generate_model(args)
    model.to(device)
    if args.is_train:
        if args.modality == 'RGB':
            summary(model, (3, args.clip_len, 112, 112))
        elif args.modality == 'Depth':
            summary(model, (1, args.clip_len, 112, 112))
        elif args.modality == 'RGB-D':
            summary(model, (4, args.clip_len, 112, 112))
        model_train(model, save_dir, dataloader_train, dataloader_val)
        pdb.set_trace()
    else:
        model_test(model, model_test_dir, '{}-{}-{}.pth'.format(args.arch, args.clip_len, args.modality),
                   dataloader_test, args.n_finetune_classes)
        pdb.set_trace()
