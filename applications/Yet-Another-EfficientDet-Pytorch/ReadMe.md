# EfficientDet

## Create llod environment

```
conda create --name llod python=3.7
```

```
source activate llod
```

## Demo


```python
source activate llod

# install requirements
pip install cython
pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors matplotlib
pip install torch==1.4.0
pip install torchvision==0.5.0
 
# run the simple inference script
python efficientdet_test.py
```

## Training

### 1 Prepare your dataset

#### VOC2COCO

```python
python pascalvoc2coco.py
```

#### Dataset format

```python
# your dataset structure should be like this
datasets/
    -your_project_name/
        -train_set_name/
            -*.jpg
        -val_set_name/
            -*.jpg
        -annotations
            -instances_{train_set_name}.json
            -instances_{val_set_name}.json

# for example, coco2017
datasets/
    -coco2017/
        -train2017/
            -000000000001.jpg
            -000000000002.jpg
            -000000000003.jpg
        -val2017/
            -000000000004.jpg
            -000000000005.jpg
            -000000000006.jpg
        -annotations
            -instances_train2017.json
            -instances_val2017.json
```

### 3.1 Train on coco from scratch

```python
CUDA_VISIBLE_DEVICES=5 python train.py -c 0 -p voc2007 --head_only True --lr 1e-3 --batch_size 32 --log_path logs_scratch/ --saved_path logs_scratch/ --num_epochs 50
```

### 3.2 Train on voc2007 from scratch

```python
CUDA_VISIBLE_DEVICES=2 python train.py -c 0 -p voc2007 --head_only True --lr 1e-3 --batch_size 32 --log_path logs_scratch/ --saved_path logs_scratch/ --num_epochs 50
```

### 3.3 Train a custom dataset with pretrained weights (Highly Recommended)

```python
CUDA_VISIBLE_DEVICES=3 python train.py -c 0 -p voc2007 --head_only True --lr 1e-3 --batch_size 32 --load_weights weights/efficientdet-d0.pth --log_path logs/ --saved_path logs/ --num_epochs 50
```

### 6. Evaluate model performance

```python
# eval on your_project, efficientdet-d0
CUDA_VISIBLE_DEVICES=4 python coco_eval.py -p voc2007 -w /home/lgzhou/Yet-Another-EfficientDet-Pytorch/logs/voc2007/efficientdet-d0_199_4600.pth -c 0
```

### Experiment Results

| Methods         | mAP  |
| --------------- | ---- |
| Efficientdet-D0 |      |
| Efficientdet-D1 |      |
| Efficientdet-D2 |      |
| Efficientdet-D3 |      |
| Efficientdet-D4 |      |
| Efficientdet-D5 |      |
| Efficientdet-D6 |      |
| Efficientdet-D7 |      |



