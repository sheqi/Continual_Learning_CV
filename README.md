## This code base is no longer maintained and exists as a historical artifact to supplement Continual Learning Toolbox for Computer Vision Tasks. For more recent work, please see [Continual_Learning_CV](https://github.com/sheqi/Continual_Learning_CV). 
# Continual_Learning_CV (CLCV)
Continual Learning Toolbox for Computer Vision Tasks


This toolbox aims at prototyping current computer vision tasks, e.g., **human gesture recognition, action localization/detection, object detection/segmentation, and person ReID** in a **continual/lifelong learning** manner. 
It means most of the SOTAs can be updated with novel data without retraining from scratch, and at the same time, they are able to migrate from **catastrophic forgetting** problem, furthermore, the models can learn with **few-shot samples and adapt quickly** to the target domains. Since the CL strategies are quite complex and flexible, it has some intersections with recent few-shot/meta/multi-task learning work.

## OpenLORIS-Object Datasets and Benchmarks
We are testing the performances based on **OpenLORIS-Object dataset**. The basic codes are the implementation of the following paper: 

Qi She et al, 
[OpenLORIS-Object: A Robotic Vision Dataset and Benchmark for Lifelong Deep Learning](https://arxiv.org/pdf/1911.06487.pdf)
The paper has been accepted into ICRA 2020. 

## Requirements

The current version of the code has been tested with following libs:
* `pytorch 1.1.0`
* `torchvision 0.2.1`
* `tqdm 4.19.9`
* `visdom 0.1.8.9`
* `Pillow 6.2.0`

Experimental platforms:
* `Intel Core i9 CPU`
* `Nvidia RTX 2080 Ti GPU`
* `CUDA Toolkit 10.*`

Install the required the packages inside the virtual environment:
```
$ conda create -n yourenvname python=3.7 anaconda
$ source activate yourenvname
$ pip install -r requirements.txt
```

## Data Preparation
Step 1: Download data (including RGB-D images, masks, and bounding boxes) following [this instruction](https://drive.google.com/open?id=1KlgjTIsMD5QRjmJhLxK4tSHIr0wo9U6XI5PuF8JDJCo). 

Step 2: Run following scripts:
```
 python3 benchmark1.py
 python3 benchmark2.py
```

Step 3: Put train/test/validation file under `./img`. For more details, please follow `note` file under each sub-directories in `./img`.

Step 4: Generate the `.pkl` files of data.
```
 python3 pk_gene.py
 python3 pk_gene_sequence.py
```
## Quickly get hands on

You can directly use scripts on 9 algorithms and 2 benchmarks (may need to modify arguments/parameters in `.bash` files if necessary):
```
bash clutter.bash
bash illumination.bash
bash pixel.bash
bash occlusion.bash
bash sequence.bash
```

## Running Benchmark 1
Individual experiments can be run with `main.py`. Main option is:

```
python3 main.py --factor
```

which kind of experiment? (`clutter`|`illumination`|`occlusion`|`pixel`)


## Running Benchmark 2
The main option to run benchmark2 is:

```
python3 main.py --factor=sequence
```

## Running specific baseline methods

- Elastic weight consolidation (EWC): 

```
main.py --ewc --savepath=ewc
```
- Online EWC:  

```
main.py --ewc --online --savepath=ewconline
```

- Synaptic intelligence (SI): 

```
main.py --si --savepath=si
```
- Learning without Forgetting (LwF): 

```
main.py --replay=current --distill --savepath=lwf
```

- Deep Generative Replay (DGR): 

```
main.py --replay=generative --savepath=dgr
```

- DGR with distillation: 

```
main.py --replay=generative --distill --savepath=distilldgr
```

- Replay-trough-Feedback (RtF): 

```
main.py --replay=generative --distill --feedback --savepath=rtf
```

- Cumulative: 

```
main.py --cumulative=1 --savepath=cumulative
```

- Naive: 

```
main.py --savepath=naive
```


## Repository Structure
```
OpenLORISCode 
├── img
├── lib
│   ├── callbacks.py
│   ├── continual_learner.py
│   ├── encoder.py
│   ├── exemplars.py
│   ├── replayer.py
│   ├── train.py
│   ├── vae_models.py
│   ├── visual_plt.py
├── _compare.py
├── _compare_replay.py
├── _compare_taskID.py
├── data.py
├── evaluate.py
├── excitability_modules.py
├── main.py
├── linear_nets.py
├── param_stamp.py
├── pk_gene.py
├── visual_visdom.py
├── utils.py
└── README.md
```
## Citation 
Please consider citing our papers if you use this code in your research:
```
@article{she2020iros,
  title={IROS 2019 Lifelong Robotic Vision Challenge--Lifelong Object Recognition Report},
  author={She, Qi and Feng, Fan and Liu, Qi and Chan, Rosa HM and Hao, Xinyue and Lan, Chuanlin and Yang, Qihan and Lomonaco, Vincenzo and Parisi, German I and Bae, Heechul and others},
  journal={arXiv preprint arXiv:2004.14774},
  year={2020}
}
```

```
@article{she2019openlorisobject,
  title={Openlorisobject: A robotic vision dataset and benchmark for lifelong deep learning},
  author={She, Qi and Feng, Fan and Hao, Xinyue and Yang, Qihan and Lan, Chuanlin and Lomonaco, Vincenzo and Shi, Xuesong and Wang, Zhengwei and Guo, Yao and Zhang, Yimin and others},
  journal={International Conference on Robotics and Automation (ICRA)},
  year={2020}
}
```

## Acknowledgements
Parts of code were borrowed from [here](https://github.com/GMvandeVen/continual-learning). 


## Issue / Want to Contribute ? 
Open a new issue or do a pull request in case you are facing any difficulty with the code base or if you want to contribute.


