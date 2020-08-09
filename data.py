from torchvision import transforms
import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

my_transform = transforms.Compose([transforms.ToTensor()])


class MyDataset(Dataset):

    def __init__(self, batch_num, name='OpenLORIS-Object', mode='train', own_transform=None, factor='clutter'):
        batch_num += 1
        self.transform = own_transform
        self.imgs = []
        self.labels = []

        if name == 'OpenLORIS-Object':
            datapath = glob.glob('{}/{}/task{}/*'.format(factor, mode, batch_num))
            datapath = sorted([p for p in datapath if p[-1].isdigit()])
            for i in range(len(datapath)):
                temp = glob.glob(datapath[i] + '/*.jpg')
                self.imgs.extend([Image.open(x).convert('RGB').resize((224, 224)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> batch{} -{}set consisting of {} samples".format(batch_num, mode, len(self)))

        elif name == 'cifar':
            for i in range(20):
                temp = glob.glob('{}/task{}/{}/*.png'.format(mode, batch_num, i + 1))
                self.imgs.extend([Image.open(x).convert('RGB') for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> batch{} -{}set consisting of {} samples".format(batch_num, mode, len(self)))

        elif name == 'mnist':
            for i in range(10):
                temp = glob.glob('{}/task{}/{}/*.png'.format(mode, batch_num, i + 1))
                self.imgs.extend([Image.open(x).convert('RGB').resize((32, 32)) for x in temp])
                self.labels.extend([i] * len(temp))
            print("  --> batch{} -{}set consisting of {} samples".format(batch_num, mode, len(self)))

    def __setitem__(self, index, value):
        self.imgs[index] = value[0]
        self.labels[index] = value[1]

    def __getitem__(self, index):

        fn = self.imgs[index]
        label = self.labels[index]
        img = fn

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):

        return len(self.imgs)


# ----------------------------------------------------------------------------------------------------------#


class SubDataset(Dataset):

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


class ExemplarDataset(Dataset):

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)


def get_multitask_experiment(name, tasks, only_config=False, factor='clutter'):
    classes_per_task = 0
    config = {}
    if name == 'OpenLORIS-Object':
        tasks = 9
        classes_per_task = 69
        config = {'size': 224, 'channels': 3, 'classes': 69}

    elif name == 'cifar':
        tasks = 5
        classes_per_task = 20
        config = {'size': 32, 'channels': 3, 'classes': 20}

    elif name == 'mnist':
        tasks = 5
        classes_per_task = 2
        config = {'size': 32, 'channels': 3, 'classes': 2}

    train_datasets = []
    test_datasets = []
    for i in range(tasks):
        train_datasets.append(MyDataset(i, name=name, mode='train', own_transform=my_transform, factor=factor))
        test_datasets.append(MyDataset(i, name=name, mode='test', own_transform=my_transform, factor=factor))

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)


'''
import pickle
with open('mnist.pk','wb') as f:
    pickle.dump(get_multitask_experiment('mnist', 1),f)
'''
