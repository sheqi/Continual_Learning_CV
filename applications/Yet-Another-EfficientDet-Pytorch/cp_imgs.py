# Author: Liguang Zhou
# Date: Aug 21. 2019

import os
import shutil


# import pandas as pd

# create a directory
def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


# cp imgs of a certain cls from file_dir to dst_dir
def cp_imgs_with_tags(cls_name, cls_num, file_dir, dst_dir, score_thre=5):
    # cls need to be changed
    # create the filename for saving files
    file_list = os.listdir(file_dir)
    print(file_list)
    print('len:', len(file_list))

    # new directory if needed
    dst_path = os.path.join(dst_dir, cls_name)
    mkdir(dst_path)

    # open CSV file
    csv_path = 'D:\\Code\\nima.pytorch-py3.7\\nima\\iqa_results.csv'
    ava_df = pd.read_csv(csv_path, sep=',')

    img_id = ava_df.iloc[:, 0]
    img_score = ava_df.iloc[:, 4]
    t1 = ava_df.iloc[:, 6]
    t2 = ava_df.iloc[:, 7]

    dst_cls_id = img_id[(t1 == cls_num) | (t2 == cls_num)]
    # dst_cls_img_score = img_score[(t1==cls_num)|(t2==cls_num)]

    dst_cls_id_high = dst_cls_id[img_score > score_thre]
    # dst_cls_img_score_high = dst_cls_img_score[img_score>score_thre]

    dst_cls_id_list = dst_cls_id_high
    print('dst_cls_id_list:', dst_cls_id_list)

    for dst_img in dst_cls_id_list:
        dst_img = str(dst_img) + '.jpg'
        idx = file_list.index(dst_img)
        print(file_list[idx])
        if os.path.exists(os.path.join(file_dir, dst_img)):
            shutil.copy(os.path.join(file_dir, dst_img), dst_path)


# cp imgs from one dir to another dir
def cp_imgs_from_files(filename_lst, img_dir, dst_dir):
    for filename in filename_lst:
        imgname = filename.strip()
        img_path = os.path.join(img_dir, imgname + '.jpg')
        dst_path = os.path.join(dst_dir, imgname + '.jpg')

        print('imgname:', imgname)
        print('imgpath:', img_path)
        print('dst_path:', dst_path)

        shutil.copy(img_path, dst_path)


# main function
if __name__ == '__main__':
    train_set_list = ['trainvalStep0', 'trainvalStep1', 'trainvalStep2', 'trainvalStep3']
    test_set_list = ['testStep0', 'testStep1', 'testStep2', 'testStep3']

    # src file dir and dst file dir
    img_dir = '/data/voc2007/JPEGImages'
    train_dst_dir = '/data/voc2007/trainvalStep3'
    test_dst_dir = '/data/voc2007/testStep3'

    train_file = '/data/voc2007/ImageSets/Main/trainvalStep3.txt'
    test_file = '/data/voc2007/ImageSets/Main/testStep3.txt'

    train_filename_file = open(train_file, "r")
    train_filename_lst = train_filename_file.readlines()

    test_filename_file = open(test_file, "r")
    test_filename_lst = test_filename_file.readlines()

    # cp the file in the filename_lst from src to destination
    cp_imgs_from_files(train_filename_lst, img_dir, train_dst_dir)
    cp_imgs_from_files(test_filename_lst, img_dir, test_dst_dir)
