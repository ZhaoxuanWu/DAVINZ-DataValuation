# DAVINZ: Data Valuation using Deep Neural Networks at Initialization [ICML-2022]

This repository is the official implementation of the following paper accepted by the Thirty-ninth International Conference on Machine Learning (ICML) 2022:

> Zhaoxuan Wu, Yao Shu, Bryan Kian Hsiang Low
>
> DAVINZ: Data Valuation using Deep Neural Networks at Initialization

## Requirements

To install requirements:
```setup
conda env create -f environment.yml
```

## Preparing datasets

*MNIST and CIFAR-10*: The code automatically downloads the required datasets. 

*MNISTM*: It can be downloaded [here](https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz). Then, place the extracted `keras_mnistm.pkl` file under the [data/](data/) folder.

*Ising Phyicial Model Dataset*: It can be downloaded at [here](https://github.com/millskyle/extensive_deep_neural_networks/blob/master/data/ising_data.h5). Then, place the `ising_data.h5` file under the [data/](data/) directory.

## Run DAVINZ baseline experiments
At the beginning of the `main.py` and `main_reg.py` files, you can find example usages of DAVINZ for classficiation and regression tasks, respectively.

We give one example here:
```bash
mkdir data results checkpoints 
python main.py --dataset=MNIST_baseline --model=ResNet18 --num_parties=10 --split_method=by_class --ground-truth --seed=0 --gpu=0
```

## Other methods
We implemented validation performance (VP), influence function (IF) and robust volume (RV) for comparisons. The code, including the example usages, can be found under the [baselines/](baselines/) directory.