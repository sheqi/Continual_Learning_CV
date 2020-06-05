import numpy as np
import struct
from PIL import Image
import matplotlib.pyplot as plt
import os


class DataUtils(object):
    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath
        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

        self._imgNums = 0
        self._LabelNums = 0

    def getImage(self):
        binfile = open(self._filename, 'rb')
        buf = binfile.read()
        binfile.close()
        index = 0
        numMagic, self._imgNums, numRows, numCols = struct.unpack_from(self._fourBytes2, buf, index)
        index += struct.calcsize(self._fourBytes)
        images = []
        print('image nums: %d' % self._imgNums)
        for i in range(self._imgNums):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j] > 1:
                    imgVal[j] = 1
            images.append(imgVal)
        return np.array(images), self._imgNums

    def getLabel(self):
        binFile = open(self._filename, 'rb')
        buf = binFile.read()
        binFile.close()
        index = 0
        magic, self._LabelNums = struct.unpack_from(self._twoBytes2, buf, index)
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(self._LabelNums):
            im = struct.unpack_from(self._labelByte2, buf, index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def outImg(self, arrX, arrY, imgNums):
        output_txt = self._outpath + '/img.txt'
        m, n = np.shape(arrX)
        for i in range(imgNums):
            img = np.array(arrX[i])
            img = img.reshape(28, 28) * 255
            outfile = 'task{}/{}/{}.png'.format(arrY[i] // 2 + 1, arrY[i] % 2 + 1, i)
            img = Image.fromarray(np.uint8(img))
            img.save(self._outpath + '/' + outfile)


if __name__ == '__main__':
    trainfile_X = 'train-images-idx3-ubyte'
    trainfile_y = 'train-labels-idx1-ubyte'
    testfile_X = 't10k-images-idx3-ubyte'
    testfile_y = 't10k-labels-idx1-ubyte'

    for i in range(10):
        os.makedirs('./train/task{}/{}/'.format(i // 2 + 1, i % 2 + 1), exist_ok=True)
        os.makedirs('./test/task{}/{}/'.format(i // 2 + 1, i % 2 + 1), exist_ok=True)

    # 加载mnist数据集
    train_X, train_img_nums = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X, test_img_nums = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    # 以下内容是将图像保存到本地文件中
    path_trainset = "train"
    path_testset = "test"
    os.makedirs(path_trainset, exist_ok=True)
    os.makedirs(path_testset, exist_ok=True)
    DataUtils(outpath=path_trainset).outImg(train_X, train_y, train_img_nums)
    DataUtils(outpath=path_testset).outImg(test_X, test_y, test_img_nums)
