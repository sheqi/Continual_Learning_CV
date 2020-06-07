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
from tensorboardX import SummaryWriter


from models import resnext 
from model import generate_model
import utils
from dataset import dataset_config, dataset_pkl, dataset_pkl_EgoGesture
from dataset.transforms import *
from dataset.temporal_transforms import *


import warnings
import os
# os.environ['CUDA_VISIBLE_DEVICES']='3'
warnings.filterwarnings("ignore")

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='0,1,2,3')

    # args for dataloader
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--batch_size', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--w', type=int, default=224)
    parser.add_argument('--h', type=int, default=224)
    parser.add_argument('--sample_size', type=int, default=224)

    parser.add_argument('--clip_len', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='jester')
    
    
    
    # args for generating the model
    parser.add_argument('--model', type=str, default='resnext')
    parser.add_argument('--arch', type=str, default='resnext-101')
    parser.add_argument('--model_depth', type=int, default=101)
    parser.add_argument('--resnet_shortcut', type=str, default='B')
    parser.add_argument('--resnext_cardinality', type=int, default=32)
    # parser.add_argument('--sample_duration', type=int, default=32)
    parser.add_argument('--pretrain_path', type=str,
    default = '/home/data2/zhengwei/R3D_pretrained_kinetics/resnext-101-kinetics.pth',
    # default='models/pretrained_models/resnext-101-kinetics.pth',
    # default = 'models/pretrained_models/resnet-50-kinetics.pth',
    help='Pretrained path for training model, DO NOT use for testing. Testing trained path is \
    defined in the main script')
    parser.add_argument('--modality', type=str, default='RGB', 
                        help='Modality of input data. RGB, Depth, RGB-D and fusion. Fusion \
                            is only used when testing the two steam model')
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
    parser.add_argument('--lr_steps', type=list, default=[10,20],
                        help='lr steps for decreasing learning rate')    
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')  
    parser.add_argument('--log', default='log', type=str)
    args = parser.parse_args()
    return args

args = parse_opts()

# ROOT_DATA_PATH = '/home/data2/zhengwei/{}'.format(args.dataset)
# annot_path = os.path.join(ROOT_DATA_PATH,'{}_annotation'.format(args.dataset))
annot_path = '{}_annotation'.format(args.dataset)
label_path = '/home/data2/zhengwei/{}/'.format(args.dataset) # for submitting testing results
# label_path = # label_path = '/home/data2/zhengwei/sth-sth-v2'


os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
device = 'cuda:0'
if isinstance(args.cuda_id, list):
    device_ids = [i for i in eval(args.cuda_id)]
else:
    device_ids = [eval(args.cuda_id)]


params = dict()
params['save_path'] = '{}-{}'.format(args.dataset, args.arch)
params['display'] = 10



def forward(model, data):
    if args.dataset == 'EgoGesture':
        rgbs, depths, labels = data
        if args.modality == 'RGB':
            inputs = rgbs.to(device, non_blocking=True).float()
        elif args.modality == 'Depth':
            inputs = depths.to(device, non_blocking=True).float()
        elif args.modality == 'RGB-D':
            inputs = torch.cat((rgbs, depths), 1).to(device, non_blocking=True).float()
    else:
        rgbs, labels = data
        inputs = rgbs.to(device, non_blocking=True).float().transpose(2,1)
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


def model_train(model, dataloader_train, dataloader_val):
    model.train()
    num_epochs = 50
    criterion = nn.CrossEntropyLoss().to(device)

    # determine optimizer
    fc_lr_layers = list(map(id, model.module.fc.parameters()))
    pretrained_lr_layers = [p for p in model.parameters() 
                            if id(p) not in fc_lr_layers and p.requires_grad==True]
    # pretrained_lr_layers = filter(lambda p: 
    #                               id(p) not in fc_lr_layers, model.parameters())
    # optimizer = torch.optim.SGD([
    #     {"params": model.module.fc.parameters()},
    #     {"params": pretrained_lr_layers, "lr": 1e-4, 'weight_decay':1e-3}
    # ], lr=1e-3, momentum=0.9, weight_decay=1e-3)    

    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)  

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    logdir = os.path.join(args.log, cur_time)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)


    # train_logger = utils.Logger(os.path.join(save_dir, '{}-{}-{}.log'.format(args.arch, args.clip_len, args.modality)),
    #                             ['step', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
    #                             'lr_feature', 'lr_fc'])
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    


    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()


    for epoch in trange(num_epochs):  # loop over the dataset multiple times
        batch_time.reset()
        data_time.reset()
        losses.reset()
        top1.reset()
        top5.reset()

        end = time.time()
        step = 0
        for data in dataloader_train:
            data_time.update(time.time() - end)
            probs, outputs, labels = forward(model, data)
            optimizer.zero_grad()
            loss_ = criterion(outputs, labels)
            loss_.backward()
            optimizer.step()
            prec1, prec5 = utils.accuracy(outputs.data, labels, topk=(1, 5))
            top1.update(prec1.item(), data[0].size(0))
            top5.update(prec5.item(), data[0].size(0))
            losses.update(loss_.item())
            batch_time.update(time.time() - end)
            end = time.time()
            if (step+1) % params['display'] == 0:
                # print('-------------------------------------------------------')
                # print('epoch{}/{} train_acc:{:.3f} train_loss:{:.3f} '.format(
                #     epoch + 1, num_epochs,
                #     train_acc.val, train_loss.val
                #     ))
                print('-------------------------------------------------------')
                for param in optimizer.param_groups:
                    print('lr: ', param['lr'])
                print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step+1, len(dataloader_train))
                print(print_string)
                print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                    data_time=data_time.val,
                    batch_time=batch_time.val)
                print(print_string)
                print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
                print(print_string)
                print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                    top1_acc=top1.avg,
                    top5_acc=top5.avg)
                print(print_string)
            step += 1     
        utils.save_checkpoint(model, optimizer, step, save_dir,
                                '{}-{}-{}-{}.pth'.format(args.arch, args.clip_len, args.modality, epoch))
        # scheduler.step()
        utils.adjust_learning_rate(args.learning_rate, optimizer, epoch, args.lr_steps)


        writer.add_scalar('train_loss_epoch', losses.avg, epoch)
        writer.add_scalar('train_top1_acc_epoch', top1.avg, epoch)
        writer.add_scalar('train_top5_acc_epoch', top5.avg, epoch)


        batch_time.reset()
        data_time.reset()
        losses.reset()
        top1.reset()
        top5.reset()

        end = time.time()
        model.eval()
        with torch.no_grad():
            for data_val in dataloader_val:
                data_time.update(time.time() - end)
                probs_val, outputs_val, labels_val = forward(model, data_val)
                val_loss_ = criterion(outputs_val, labels_val)
                losses.update(val_loss_.item())
                prec1, prec5 = utils.accuracy(outputs.data, labels, topk=(1, 5))
                top1.update(prec1.item(), data[0].size(0))
                top5.update(prec5.item(), data[0].size(0))
                batch_time.update(time.time() - end)
                end = time.time()
        model.train()
        print('----validation----')
        print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(dataloader_val))
        print(print_string)
        print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
            data_time=data_time.val,
            batch_time=batch_time.val)
        print(print_string)
        print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
        print(print_string)
        print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
            top1_acc=top1.avg,
            top5_acc=top5.avg)
        print(print_string)


        writer.add_scalar('val_loss_epoch', losses.avg, epoch)
        writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
        writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)






if __name__ == '__main__':
    # keep shuffling be constant every time
    seed = 1
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    mean=[.485, .456, .406]
    std=[.229, .224, .225]

    scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        scales.append(scales[-1] * args.scale_step)



    temporal_transform_train = transforms.Compose([
            TemporalUniformCrop(args.clip_len)
            ])    

    temporal_transform_test = transforms.Compose([
            TemporalUniformCrop(args.clip_len)
            ])

    trans_train  = transforms.Compose([
                            GroupScale([256, 256]),  #112/0.8
                            GroupMultiScaleCrop([args.w, args.h], scales),
                            # GroupMultiScaleRotate(20),
                            ToTorchFormatTensor(),
                            Stack_3D(),
                            GroupNormalize(mean=mean, std=std)])


    trans_test  = transforms.Compose([
                           GroupScale([args.w, args.h]),
                           ToTorchFormatTensor(),
                           Stack_3D(),
                           GroupNormalize(mean=mean, std=std)])

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
    params['num_classes'] = num_class

    # load dataset
    if args.is_train:
        print('Loading training data.....')
        if args.dataset == 'EgoGesture':
            dataset_train = dataset_pkl_EgoGesture.dataset_video(annot_path, 'train',
                                                spatial_transform=trans_train,
                                                temporal_transform = temporal_transform_train)
            dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                            shuffle=True, 
                                            num_workers=args.num_workers, pin_memory=True)

            print('\n')
            print('Loading validating data.....')
            dataset_val = dataset_pkl_EgoGesture.dataset_video(annot_path, 'val', 
                                                spatial_transform=trans_test,
                                                temporal_transform = temporal_transform_test)
            dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, 
                                        num_workers=args.num_workers,pin_memory=True)
        else:
            dataset_train = dataset_pkl.dataset_video(annot_path, 'train',
                                                spatial_transform=trans_train,
                                                temporal_transform = temporal_transform_train)
            dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                            shuffle=True, 
                                            num_workers=args.num_workers, pin_memory=True)

            print('\n')
            print('Loading validating data.....')
            dataset_val = dataset_pkl.dataset_video(annot_path, 'val', 
                                                spatial_transform=trans_test,
                                                temporal_transform = temporal_transform_test)
            dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, 
                                        num_workers=args.num_workers,pin_memory=True)            


        
    else:
        print('Loading testing data.....')
        if args.dataset == 'EgoGesture':
            dataset_test = dataset_pkl_EgoGesture.dataset_video(annot_path, 'test', 
                                                spatial_transform=trans_test,
                                                temporal_transform = temporal_transform_test)
            dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, 
                                        num_workers=args.num_workers,pin_memory=True)
        else:
            dataset_test = dataset_pkl.dataset_video(annot_path, 'test', 
                                                spatial_transform=trans_test,
                                                temporal_transform = temporal_transform_test)
            dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, 
                                        num_workers=args.num_workers,pin_memory=True)

    model, parameters = generate_model(args, params['num_classes'])
    model.to(device)



    if args.is_train:
        if args.modality == 'RGB':
            summary(model, (3,args.clip_len,args.h,args.h))
        elif args.modality == 'Depth':
            summary(model, (1,args.clip_len,args.h,args.h))
        elif args.modality == 'RGB-D':
            summary(model, (4,args.clip_len,args.h,args.h))
        model_train(model, dataloader_train, dataloader_val)
        pdb.set_trace()
    else:
        # to be fixed for submitting testing results for jester and sthv2
        model_test(model, params['save_path'], '{}-{}-{}.pth'.format(args.arch, args.clip_len, args.modality), dataloader_test, args.n_finetune_classes)
        pdb.set_trace()