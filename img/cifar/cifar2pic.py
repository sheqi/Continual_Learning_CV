import sys
import cv2
import numpy as np
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


for i in range(100):
    os.makedirs('./train/task{}/{}/'.format(i//20+1,i%20+1),exist_ok=True)
    os.makedirs('./test/task{}/{}/'.format(i//20+1,i%20+1),exist_ok=True)

def toimg(num,dirname,dataset):
    idx=np.argsort(dataset[b'coarse_labels'])
    filenames=[x.decode() for x in dataset[b'filenames']]
    filenames=np.array(filenames)[idx]
    data=np.array(dataset[b'data'])[idx]
    fine=np.array(dataset[b'fine_labels'])[idx]
    coarse=np.array(dataset[b'coarse_labels'])[idx]
    gap=num//20
    for c in range(20):
        subidx=np.argsort(fine[c*gap:(c+1)*gap])
        subdata=data[c*gap:(c+1)*gap][subidx]
        subnames=filenames[c*gap:(c+1)*gap][subidx]
        subgap=num/100
        for i in range(gap):
            img=np.reshape(subdata[i],(3, 32, 32))
            img=img[[2,1,0]].transpose(1, 2, 0) 
            name='{}/task{}/{}/{}'.format(dirname,i//(num//100)+1,c+1,subnames[i])
            cv2.imwrite(name,img)

toimg(10000,'test',unpickle('cifar-100-python/test'))
toimg(50000,'train',unpickle('cifar-100-python/train'))
