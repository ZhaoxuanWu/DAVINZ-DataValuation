"""
    Example usage:
    
    # Correlation experiment
    python main.py --dataset=MNIST --model=CNN --num_parties=200 --split_method=correlation_exp --ground-truth --no-loo
    
    # Baselines
    python main.py --dataset=MNIST_baseline_resized --model=VGG13 --num_parties=10 --split_method=by_class --ground-truth --seed=0 --gpu=0
    python main.py --dataset=MNIST_baseline --model=ResNet18 --num_parties=10 --split_method=by_class --ground-truth --seed=0 --gpu=0
    python main.py --dataset=CIFAR_10_baseline --model=VGG13 --num_parties=10 --split_method=by_class --ground-truth --seed=0 --gpu=0
    python main.py --dataset=CIFAR_10_baseline --model=ResNet18 --num_parties=10 --split_method=by_class --ground-truth --seed=0 --gpu=0
    
    # Awareness of data preference: valuation with domain shift
    python main.py --dataset=MNIST_MNISTM --model=CNN --num_parties=10 --split_method=val_domain_shift --trim_dataset=10000 --ground-truth --seed=0 --gpu=0 --no-loo
    
    # Robustness to model
    python main.py --dataset=CIFAR_10 --model=ResNet18 --num_parties=10 --split_method=random --trim_dataset=10000 --get-eigen --seed=0 --gpu=0
    python main.py --dataset=CIFAR_10 --model=ResNet34 --num_parties=10 --split_method=random --trim_dataset=10000 --get-eigen --seed=0 --gpu=0
"""

import time
import os
import argparse
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from mmd import rbf_mmd2
from ntk import compute_ntk_score_batched, compute_ntk_score_batched_permute
from utils import load_dataset_cls, split_dataset_for_MNIST_MNISTM, train, test

from model.cnn import CNN
from model.resnet import BasicBlock, ResNet
from model.vgg import VGG

parser = argparse.ArgumentParser(description='Data valuation at initialization (classification task).')
parser.add_argument('--gpu', help='gpu device index',
                    required=False,
                    type=str,
                    default='0')
parser.add_argument('--seed', help='seed for reproducibility',
                    required=False,
                    type=int,
                    default=0)
parser.add_argument('--dataset', help='dataset to use: MNIST, MNIST_baseline, MNIST_MNISTM, CIFAR_10, CIFAR_10_baseline ',
                    required=True,
                    type=str,
                    default='MNIST')
parser.add_argument('--trim_dataset', help='number of data points to keep for faster training and experiment',
                    required=False,
                    type=int,
                    default=10000)
parser.add_argument('--num_parties', help='number of parties to value',
                    required=True,
                    type=int,
                    default=10)
parser.add_argument('--split_method', help='method to split the dataset: by_class, random, correlation_exp, val_domain_shift',
                    required=False,
                    type=str,
                    default='by_class')
parser.add_argument('--model', help='model to use: CNN, VGG11, VGG13, VGG16, ResNet18, ResNet21, ResNet34',
                    required=True,
                    type=str,
                    default='CNN')
parser.add_argument('-loo', help='whether to compute leave-one-out value',
                    required=False,
                    type=bool,
                    default=True)
parser.add_argument('--no-loo', dest='loo', 
                    action='store_false')
parser.add_argument('--get-eigen', help='whether to calculate eigenvalue of ntk',
                    action='store_true')
parser.add_argument('--ground-truth', help='retrain models explicitly till convergence to get ground truth',
                    action='store_true')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    # Reporducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    
def get_indices_on_split(train_images, train_labels, party_i, split, extra_args={}, loo=True):
    """
    Given the data, party index and split method name, return the indices of data samples concerned.
    
    Args:
        train_images (Tensor): the training images
        train_labels (Tensor): corresponding labels of the training images
        party_i (int): party index that is concerned
        split (str): a str specifying the type of split
        extra_args (dict): a dictionary containing extra args required for some specific split types
        loo (bool): if True, return the data sample indices excluding the data from party i
    Return:
        (np.array) the indices of data sample from a party i (or the complement if loo==True)
    """
    size = train_labels.shape[0]
    if split == 'by_class':
        # Each party should only contain data of a single class
        party_i_indices = (train_labels == party_i).nonzero()[0]
    elif split == 'random':
        # Randomly permute the whole dataset before splitting them to datasets
        permutation_indices = np.random.RandomState(seed=0).permutation(size)
        each_party = size//extra_args['num_parties']
        if party_i == extra_args['num_parties']:
            party_i_indices = []
        else:
            party_i_indices = permutation_indices[np.arange(party_i * each_party, (party_i + 1) * each_party)]
    elif split == 'correlation_exp':
        # For the correlation exp in Sec 6.1, bootstrap datasets of size up to 10000
        party_i_size = np.random.randint(1, 10000)
        party_i_indices = np.random.randint(0, size, party_i_size)
    elif split == 'val_domain_shift':
        # For the awareness to data preference experiment in Sec 6.3
        party_i_indices = [party_i]
    else:
        raise NotImplementedError()
    if loo:
        # Set difference for leave-one-out
        if party_i == extra_args['num_parties']:
            party_i_indices = []
        coalition_i_indices = np.setdiff1d(np.arange(0, size), party_i_indices)
    else:
        coalition_i_indices = party_i_indices
    return coalition_i_indices


def main():
    # Load the datasets
    train_images, train_labels, test_images_tensor, test_labels, dims, num_to_keep = load_dataset_cls(args.dataset, args.trim_dataset, args.num_parties)
    args.trim_dataset = num_to_keep
    mode = 'cls'
    
    # Reproducibility
    set_seed(args.seed)
    
    # Construct NN model
    in_channels = dims[0]
    retrain_threshold = 1e-8 # Note this retrain is different from VP, it finds the ground truth by training till convergence
    lr = 0.01
    diagonal_I_mag = 1e-4
    
    if args.get_eigen:
        # For more accurate eigen computation in conditional model agnostic experiment,
        # we set a smaller diagonal correction term
        diagonal_I_mag = 1e-6
    
    # Set the batch numbers for the diagonal approximation of NTK matrices
    ntk_n_batch = 1 if args.model == 'CNN' else 100
    
    if args.model == 'CNN':
        if 'CIFAR_10' in args.dataset:
            model = CNN(in_channels=in_channels, linear_dim=dims[2]//2//2).to(device)
            retrain_threshold = 1e-6
        else:
            model = CNN(in_channels=in_channels).to(device)
            retrain_threshold = 1e-6
    elif args.model == 'ResNet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels).to(device)
    elif args.model == 'ResNet21':
        model = ResNet(BasicBlock, [2, 3, 3, 3], in_channels=in_channels).to(device)
    elif args.model == 'ResNet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels).to(device)
    elif args.model == 'VGG11':
        model = VGG('VGG11', in_channels=in_channels).to(device)
    elif args.model == 'VGG13':
        model = VGG('VGG13', in_channels=in_channels).to(device)
    elif args.model == 'VGG16':
        model = VGG('VGG16', in_channels=in_channels).to(device)
    else:
        raise NotImplementedError()
    init_path = 'checkpoints/{}_{}_init_seed{}.pt'.format(args.dataset.lower(), args.model.lower(), args.seed)
    torch.save(model, init_path)
    
    # DaVinz calculations
    MMDs = []
    ntk_scores = []
    combined_scores = []
    accuracies = []
    losses = []
    min_eigens = []
    davinz_times = []
    retrain_times = []

    for i in range(args.num_parties + (1 if args.loo else 0)): # Last iteration for grand coalition of datasets in LOO
        
        print('Evaluating the value of party {} ...'.format(i))
        
        # Get indices for data points leaving out party i
        extra_args = {
            'num_parties': args.num_parties,
        }
        coalition_i_indices = get_indices_on_split(train_images, train_labels, 
                                                   party_i=i, split=args.split_method, 
                                                   extra_args=extra_args, loo=args.loo)
        coalition_i_images_tensor = torch.tensor(train_images[coalition_i_indices], dtype=torch.float32)
        coalition_i_labels_tensor = torch.tensor(train_labels[coalition_i_indices])
        davinz_start_time = time.time()
        
        # MMD (out-of-domain generalization error)
        mmd_squared = rbf_mmd2(coalition_i_images_tensor.reshape(-1, np.prod(dims)), 
                               test_images_tensor.reshape(-1, np.prod(dims)).float(), 
                               sigma=5)
        mmd = torch.sqrt(mmd_squared)
        MMDs.append(mmd)
        print('MMD: {}'.format(mmd))

        # NTK (in-domain generalization error)
        model = torch.load(init_path).to(device)
        inputs = coalition_i_images_tensor.reshape(-1, *dims)
        coalition_i_labels_tensor = coalition_i_labels_tensor.reshape(-1)
        score, min_eigen = compute_ntk_score_batched_permute(model, inputs.to(device), 
                                                             coalition_i_labels_tensor.to(device), 
                                                             mode, n_batch=ntk_n_batch, n_permute=1, 
                                                             diagonal_I_mag=diagonal_I_mag, get_eigen=args.get_eigen)
        ntk_scores.append(score)
        min_eigens.append(min_eigen)
        print('NTK: {}'.format(score))
        
        davinz_times.append(time.time() - davinz_start_time)
        print('DaVinz time: {}'.format(davinz_times[-1]))
                
        # Get the ground truth by retaining till convergence
        if args.ground_truth:
            retrain_start_time = time.time()
            coalition_i_data = TensorDataset(inputs.to(device), coalition_i_labels_tensor.to(device))
            test_data = TensorDataset(test_images_tensor.float().to(device), torch.tensor(test_labels).to(device))
            
            loaders = {
                'train' : torch.utils.data.DataLoader(coalition_i_data, 
                                                      batch_size=128, 
                                                      shuffle=True),

                'test'  : torch.utils.data.DataLoader(test_data, 
                                                      batch_size=128, 
                                                      shuffle=False),
            }
            model = torch.load(init_path).to(device)
            loss_func = torch.nn.CrossEntropyLoss()   
            optimizer = optim.SGD(model.parameters(), lr = lr) 
            
            train(model, loaders, loss_func, optimizer, device, num_epochs=1000, threshold=retrain_threshold, minimum_epochs=50)
            accuracy, loss = test(model, loaders, loss_func, device)
            accuracies.append(accuracy)
            losses.append(loss)
            print('Accuracy: ', accuracy)
            retrain_times.append(time.time() - retrain_start_time)
            print('Retrain time: {}'.format(retrain_times[-1]))
    
    # Combining the in-domain and out-of-domain scores (aka v(S))
    # Note: a better alternative is to average across different runs/seeds - Eqn (4) of paper
    kappa = np.mean(MMDs)/ np.mean(ntk_scores)
    combined_scores = - kappa * np.array(ntk_scores) - np.array(MMDs)
    print('Combined scores: ', combined_scores)
    
    np.savez('results/{}_{}_{}_{}_kept{}_seed{}{}.npz'.format(args.dataset, args.model, args.num_parties, args.split_method, args.trim_dataset, args.seed, '_retrained' if args.ground_truth else ''),
             MMDs=MMDs,
             ntk_scores=ntk_scores,
             combined_scores=combined_scores,
             accuracies=accuracies,
             losses=losses,
             min_eigens=min_eigens,
             davinz_times=davinz_times,
             retrain_times=retrain_times)

if __name__ == '__main__':
    main()
